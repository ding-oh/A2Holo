"""Causal sensitivity of each block: scale its residual update, h_in + alpha * (h_out - h_in) (structure stream
only in a double block), and write the change in Cα RMSD of x_apo + v_ca to holo per block and per pair.

    python analysis/run.py --script analysis/alpha_scaling_reference_set.py --t 0.5 \\
        --checkpoint seed1=checkpoints/a2holo_seed1 --max_pairs 40 --tag alpha_scaling_reference_set
"""
import os, sys, gc, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, test_pairs, load_pair, MODELS, T_VAL, MAX_LEN, OUT_DIR as OUT, kabsch_rmsd  # noqa: E402

# alpha = 0 removes the block's contribution; 0.5 is one intermediate dose.
ALPHAS = (0.0, 0.5)


class Damper:
    """Scale one block's residual update by alpha; identity when disabled. `applied` counts applications."""

    def __init__(self, model):
        self.blocks = list(model.double_blocks) + list(model.single_blocks)
        self.n_double = len(model.double_blocks)
        self.target, self.alpha = None, 1.0
        self.applied = 0
        self._hooks = []

    def register(self):
        # Double blocks take keyword arguments only, so the hook needs with_kwargs=True to see its input.
        for i, b in enumerate(self.blocks):
            self._hooks.append(
                b.register_forward_hook(self._hook(i), with_kwargs=True))

    def _hook(self, i):
        def hook(module, args, kwargs, output):
            if self.target != i or self.alpha == 1.0:
                return None
            if i < self.n_double:
                struct_in = kwargs["struct"]
                struct_out, seq_out = output
                self.applied += 1
                return (struct_in + self.alpha * (struct_out - struct_in), seq_out)
            x_in = args[0]
            x_out = output[0] if isinstance(output, tuple) else output
            self.applied += 1
            return x_in + self.alpha * (x_out - x_in)
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=40)
    ap.add_argument("--tag", default="alpha_scaling_reference_set")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.join(OUT, args.tag)), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = test_pairs()
    print(f"[env] device={device}  pairs={len(pairs)}  alphas={ALPHAS}", flush=True)

    rows = []
    perpair = []
    for label, ckdir in MODELS:
        path = os.path.join(ckdir, "best_model.pt")
        if not os.path.exists(path):
            continue
        model, n_double = load_model(path, device)
        d = Damper(model)
        d.register()
        n_blocks = len(d.blocks)
        print(f"[model] {label:<18s} blocks={n_blocks} (double={n_double})", flush=True)

        base, dmg = [], {(li, a): [] for li in range(n_blocks) for a in ALPHAS}
        used = 0
        for pid in pairs:
            if used >= args.max_pairs:
                break
            p = load_pair(pid, device)
            if p["n"] > MAX_LEN:
                continue
            N, ca0, caT = p["n"], p["ca0"], p["caT"]
            pca = (1 - T_VAL) * ca0 + T_VAL * caT
            pcb = (1 - T_VAL) * p["cb0"] + T_VAL * p["cbT"]
            t = torch.full((1,), T_VAL, device=device)

            def run():
                with torch.no_grad():
                    v_ca, _ = model(p["x"], pca, pcb, t, p["bidx"], p["seq_emb"], p["global_token"])
                return kabsch_rmsd((ca0 + v_ca).double(), caT.double())

            d.target = None; d.alpha = 1.0
            b = run(); base.append(b)
            for li in range(n_blocks):
                for a in ALPHAS:
                    d.target, d.alpha = li, a
                    before = d.applied
                    dmg[(li, a)].append(run() - b)
                    if d.applied == before:
                        raise RuntimeError(f"block {li} alpha={a}: the hook never fired")
            d.target, d.alpha = None, 1.0
            for li in range(n_blocks):
                perpair.append(dict(
                    model=label, pair=pid, N=N, block=li, n_double=n_double,
                    block_type="double" if li < n_double else "single", base_rmsd=b,
                    **{f"dRMSD_a{al}": dmg[(li, al)][-1] for al in ALPHAS}))
            used += 1
            del p
            gc.collect()

        for li in range(n_blocks):
            r = dict(model=label, block=li, n_double=n_double,
                     block_type="double" if li < n_double else "single",
                     block_frac=li / max(n_blocks - 1, 1), n=used,
                     base_rmsd=float(np.mean(base)))
            for a in ALPHAS:
                r[f"dRMSD_a{a}"] = float(np.mean(dmg[(li, a)]))
            rows.append(r)
        cur = rows[-n_blocks:]
        v = [c["dRMSD_a0.0"] for c in cur]
        print(f"[{label}] base RMSD={np.mean(base):.3f} A  "
              f"ablate(a=0) dRMSD depth-mean={np.mean(v):+.3f} "
              f"range {min(v):+.3f}..{max(v):+.3f} A", flush=True)
        d.remove(); del model, d
        torch.cuda.empty_cache(); gc.collect()

    keys = ["model", "block", "n_double", "block_type", "block_frac", "n",
            "base_rmsd"] + [f"dRMSD_a{a}" for a in ALPHAS]
    tsv = os.path.join(OUT, f"{args.tag}.tsv")
    with open(tsv, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")
    print(f"[done] wrote {tsv}", flush=True)
    pp = os.path.join(OUT, f"{args.tag}_perpair.tsv")
    pkeys = list(perpair[0]) if perpair else []
    with open(pp, "w") as f:
        f.write("\t".join(pkeys) + "\n")
        for r in perpair:
            f.write("\t".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                              for k in pkeys) + "\n")
    print(f"[done] wrote {pp}  ({len(perpair)} pair x block rows)", flush=True)


if __name__ == "__main__":
    main()
