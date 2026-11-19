"""Relative decodability E-tilde per block: held-out ridge-probe error of the structure-stream state over
that of the sequence embedding alone (< 1: the block carries transition information beyond sequence).

    python analysis/run.py --script analysis/decodability_relative_reference_set.py --t 0 \\
        --checkpoint seed1=checkpoints/a2holo_seed1 --max_pairs 150 --tag t0/decodability_relative_reference_set
"""
import os, sys, gc, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, test_pairs, load_pair, MODELS, T_VAL, MAX_LEN, OUT_DIR as OUT  # noqa: E402

N_FOLDS = 5
LAMBDAS = (1e-2, 1e-1, 1.0, 1e1, 1e2)   # ridge path; the best held-out R^2 over it is kept
# Rotation-invariant targets: |x_holo,i - x_apo,i| and sum_j |D_holo,ij - D_apo,ij|. The displacement
# vector itself is not a target: the Kabsch loss leaves the output frame free per pair.
TARGETS = ("absdisp", "ddmrow")


class StateCapturer:
    """The structure-stream state leaving every block; in a single block it is the second half of [seq; struct]."""

    def __init__(self, model):
        self.model = model
        self.states = {}
        self._hooks = []
        self._L = None

    def set_L(self, L):
        self._L = L

    def register(self):
        for i, b in enumerate(self.model.double_blocks):
            self._hooks.append(b.register_forward_hook(self._double(i)))
        n_d = len(self.model.double_blocks)
        for j, b in enumerate(self.model.single_blocks):
            self._hooks.append(b.register_forward_hook(self._single(n_d + j)))

    def _double(self, idx):
        def hook(module, inputs, output):
            struct = output[0] if isinstance(output, tuple) else output
            self.states[idx] = struct.detach()[0]           # [N, d]
        return hook

    def _single(self, idx):
        def hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            L = self._L
            assert x.shape[1] == 2 * L, f"single block got {x.shape}, expected 2L={2*L}"
            self.states[idx] = x.detach()[0, L:, :]
        return hook

    def check(self):
        """Raise unless the last forward captured every double and single block."""
        n = len(self.model.double_blocks) + len(self.model.single_blocks)
        if len(self.states) != n:
            raise RuntimeError(f"captured {len(self.states)} of {n} blocks; the state hooks did not run")

    def clear(self):
        self.states = {}

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


class RidgeAccum:
    """Streaming sufficient statistics X'X, X'y for a grouped-CV ridge probe."""

    def __init__(self, d, n_folds, device):
        self.d = d
        self.A = [torch.zeros(d + 1, d + 1, dtype=torch.float64, device=device)
                  for _ in range(n_folds)]
        self.b = [torch.zeros(d + 1, dtype=torch.float64, device=device)
                  for _ in range(n_folds)]
        self.yy = [0.0] * n_folds
        self.n = [0] * n_folds

    def add(self, X, y, f):
        X = torch.cat([X.double(), torch.ones(X.shape[0], 1, dtype=torch.float64,
                                              device=X.device)], dim=1)
        y = y.double()
        self.A[f] += X.T @ X
        self.b[f] += X.T @ y
        self.yy[f] += float(y @ y)
        self.n[f] += X.shape[0]

    def cv_r2(self, lambdas):
        """Held-out R^2, best lambda over the path, in closed form:
        SSE_f = y'y - 2 w'b_f + w'A_f w with w fitted on every other fold."""
        nf = len(self.A)
        tot_y = sum(float(self.b[f][-1]) for f in range(nf))   # last column is the ones
        tot_n = sum(self.n)
        tot_yy = sum(self.yy)
        tss = tot_yy - tot_y ** 2 / max(tot_n, 1)
        best = -np.inf
        eye = torch.eye(self.d + 1, dtype=torch.float64, device=self.A[0].device)
        eye[-1, -1] = 0.0                                       # do not penalise intercept
        for lam in lambdas:
            sse = 0.0
            for f in range(nf):
                A = sum(self.A[g] for g in range(nf) if g != f)
                b = sum(self.b[g] for g in range(nf) if g != f)
                try:
                    w = torch.linalg.solve(A + lam * eye, b)
                except Exception:
                    w = torch.linalg.lstsq(A + lam * eye, b.unsqueeze(1)).solution.squeeze(1)
                sse += self.yy[f] - 2 * float(w @ self.b[f]) + float(w @ (self.A[f] @ w))
            r2 = 1.0 - sse / tss if tss > 0 else np.nan
            best = max(best, r2)
        return float(best), float(tss / max(tot_n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=150)
    ap.add_argument("--tag", default="decodability_relative_reference_set")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.join(OUT, args.tag)), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = test_pairs()
    print(f"[env] device={device}  candidate pairs={len(pairs)}  folds={N_FOLDS}", flush=True)

    rows = []
    for label, ckdir in MODELS:
        path = os.path.join(ckdir, "best_model.pt")
        if not os.path.exists(path):
            print(f"[skip] {label}: no checkpoint", flush=True)
            continue
        model, n_double = load_model(path, device)
        cap = StateCapturer(model)
        cap.register()
        n_blocks = len(model.double_blocks) + len(model.single_blocks)
        print(f"[model] {label:<18s} double={n_double} total_blocks={n_blocks}", flush=True)

        acc = {(t, li): None for t in TARGETS for li in range(n_blocks)}
        base = {t: None for t in TARGETS}          # sequence-only denominator
        used = 0
        for pid in pairs:
            if used >= args.max_pairs:
                break
            p = load_pair(pid, device)
            if p["n"] > MAX_LEN:
                continue
            N, seq_emb = p["n"], p["seq_emb"]

            a0, aT = p["data"].pos_0.double(), p["data"].pos_T.double()
            tgt = {
                "absdisp": (aT - a0).norm(dim=-1).to(device),
                "ddmrow": (torch.cdist(aT, aT) - torch.cdist(a0, a0)).abs().sum(0).to(device),
            }
            fold = used % N_FOLDS          # folds by pair: all residues of a pair share a fold, but pairs of one protein may not

            cap.clear(); cap.set_L(N)
            with torch.no_grad():
                model(p["x"], (1 - T_VAL) * p["ca0"] + T_VAL * p["caT"],
                      (1 - T_VAL) * p["cb0"] + T_VAL * p["cbT"],
                      torch.full((1,), T_VAL, device=device), p["bidx"], seq_emb, p["global_token"])
            cap.check()

            for t in TARGETS:
                y = tgt[t]
                if base[t] is None:
                    base[t] = RidgeAccum(seq_emb.shape[1], N_FOLDS, device)
                base[t].add(seq_emb, y, fold)
                for li, h in cap.states.items():
                    if acc[(t, li)] is None:
                        acc[(t, li)] = RidgeAccum(h.shape[1], N_FOLDS, device)
                    acc[(t, li)].add(h, y, fold)
            used += 1
            cap.clear()
            del p
            gc.collect()

        for t in TARGETS:
            r2_base, _ = base[t].cv_r2(LAMBDAS)
            for li in range(n_blocks):
                if acc[(t, li)] is None:
                    continue
                r2, _ = acc[(t, li)].cv_r2(LAMBDAS)
                # E-tilde = (1 - R2_block) / (1 - R2_seq); the shared TSS cancels.
                et = (1 - r2) / (1 - r2_base) if (1 - r2_base) > 1e-12 else np.nan
                rows.append(dict(model=label, target=t, block=li,
                                 block_frac=li / max(n_blocks - 1, 1),
                                 n_double=n_double, n=used,
                                 r2=r2, r2_seq_baseline=r2_base, E_tilde=et))
            cur = [r for r in rows if r["model"] == label and r["target"] == t]
            print(f"[{label}] {t:<8s} R2_seq={r2_base:+.4f}  "
                  f"R2 depth-mean={np.nanmean([r['r2'] for r in cur]):+.4f}  "
                  f"E~ depth-mean={np.nanmean([r['E_tilde'] for r in cur]):.4f}", flush=True)

        cap.remove(); cap.clear()
        del model, cap, acc, base
        torch.cuda.empty_cache(); gc.collect()

    keys = ["model", "target", "block", "block_frac", "n_double", "n",
            "r2", "r2_seq_baseline", "E_tilde"]
    tsv = os.path.join(OUT, f"{args.tag}.tsv")
    with open(tsv, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")
    print(f"[done] wrote {tsv}", flush=True)


if __name__ == "__main__":
    main()
