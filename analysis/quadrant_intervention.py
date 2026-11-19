#!/usr/bin/env python3
r"""Causal effect of cross-stream attention: in one double-stream block at a time, scale the cross (seq<->struct)
or, as a size-matched control, the intra quadrants of the joint attention map by alpha and write the change in
Cα RMSD of x_apo + v_ca to holo. Needs a GPU.

    python analysis/run.py --script analysis/quadrant_intervention.py --t 0.5 \
        --checkpoint seed1=checkpoints/a2holo_seed1 --max_pairs 40 --tag quadrant_intervention
    python analysis/run.py --script analysis/quadrant_intervention.py --t 0.5 --selftest_only
"""
import os, sys, gc, math, argparse

import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, test_pairs, load_pair, MODELS, T_VAL, MAX_LEN, OUT_DIR as OUT, kabsch_rmsd  # noqa: E402
import a2holo.model as a2holo_model  # noqa: E402

# The patch of a2holo_model.attention only reaches the blocks if they look the name up in this module's globals.
assert a2holo_model.DoubleStreamBlock.forward.__globals__ is vars(a2holo_model), \
    "DoubleStreamBlock.forward does not read attention() from a2holo.model's globals"

ALPHAS = (0.0, 0.5)
MODES = ("cross", "intra")
# Rows are renormalised to sum to 1, so alpha = 0 moves the attention mass to the other quadrant.
RENORMALISE = True
ROWSUM_TOL = 1e-3           # |sum_j A_ij - 1| after the intervention
IDENTITY_TOL = 5e-3         # alpha = 1.0 must reproduce the control, in Angstrom


def quadrant_slices(n_seq, n_struct, mode):
    """(row, col) slice pairs for one arm, token order [seq; struct] (checked by verify_token_order)."""
    s, c = slice(None, n_seq), slice(n_seq, None)       # seq half, struct half
    if mode == "cross":
        return [(s, c), (c, s)]        # seq->struct, struct->seq
    if mode == "intra":
        return [(s, s), (c, c)]        # seq->seq, struct->struct
    raise ValueError("mode must be 'cross' or 'intra', got %r" % (mode,))


def apply_quadrant(A, n_seq, n_struct, mode, alpha, renorm=RENORMALISE):
    """Scale two quadrants of a row-stochastic attention tensor A [..., S, S] by alpha, IN PLACE.

    Returns (mass_before, mass_after, rowsum_dev): the share of the total mass in the targeted quadrants
    before and after, and the worst |rowsum - 1| afterwards.
    """
    S = n_seq + n_struct
    if A.shape[-1] != S or A.shape[-2] != S:
        raise RuntimeError(
            "attention map is %s but n_seq+n_struct = %d+%d = %d"
            % (tuple(A.shape), n_seq, n_struct, S))
    sl = quadrant_slices(n_seq, n_struct, mode)
    want = {"cross": [(n_seq, n_struct), (n_struct, n_seq)],
            "intra": [(n_seq, n_seq), (n_struct, n_struct)]}[mode]
    total = A.sum()
    before = 0.0
    for (r, c), shape in zip(sl, want):
        blk = A[..., r, c]
        if tuple(blk.shape[-2:]) != shape:
            raise RuntimeError("quadrant %s slice gave %s, expected %s"
                               % (mode, tuple(blk.shape[-2:]), shape))
        before = before + blk.sum()
    for r, c in sl:
        A[..., r, c] *= alpha
    if renorm:
        A /= A.sum(-1, keepdim=True).clamp_min(1e-20)
    after = sum(A[..., r, c].sum() for r, c in sl)
    dev = (A.sum(-1) - 1.0).abs().max()
    mass_before, mass_after, rowsum_dev = torch.stack(
        [before / total, after / total, dev]).double().tolist()
    return mass_before, mass_after, rowsum_dev


class QuadrantIntervenor:
    """Replaces a2holo.model.attention with an explicit-softmax version that edits the quadrants of the
    attention map in ONE target double block; block hooks tell it which block is running, and every other
    call (other double blocks, single blocks) goes to the original attention."""

    def __init__(self, model):
        self.dblocks = list(model.double_blocks)
        self.n_double = len(self.dblocks)
        self.target, self.mode, self.alpha = None, None, 1.0
        self.applied = 0          # times the patched path ran for the target
        self.modified = 0         # times the targeted mass provably dropped
        self.n_seq = self.n_struct = None
        self.last_target_mass = float("nan")
        self.worst_rowsum_dev = 0.0
        self.probe_capture = None
        self.probe_perturb = None      # None | "struct" | "seq"
        self._cur = None
        self._hooks, self._orig = [], None

    def register(self):
        # Double blocks take keyword arguments only, so the hooks need with_kwargs=True to see them.
        for i, b in enumerate(self.dblocks):
            self._hooks.append(b.register_forward_pre_hook(
                self._pre(i), with_kwargs=True))
            self._hooks.append(b.register_forward_hook(
                self._post(), with_kwargs=True))
        self._orig = a2holo_model.attention
        a2holo_model.attention = self._attention

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        if self._orig is not None:
            a2holo_model.attention = self._orig
            self._orig = None

    def _pre(self, i):
        def hook(module, args, kwargs):
            struct, seq = kwargs["struct"], kwargs["seq"]
            self.n_seq, self.n_struct = seq.shape[1], struct.shape[1]
            self._cur = i
            if self.probe_perturb is not None and i == 0:
                kw = dict(kwargs)
                tgt = kw[self.probe_perturb]
                # A random field, not a constant: the pre-attention LayerNorm would remove a uniform offset.
                g = torch.Generator(device=tgt.device)
                g.manual_seed(20260912)
                kw[self.probe_perturb] = tgt + 3.0 * torch.randn(
                    tgt.shape, generator=g, device=tgt.device, dtype=tgt.dtype)
                return (args, kw)
            return None
        return hook

    def _post(self):
        def hook(module, args, kwargs, output):
            self._cur = None
            return None
        return hook

    def _attention(self, q, k, v, pe):
        probing = self.probe_capture is not None and self._cur == 0
        if probing:
            self.probe_capture.append(q.detach().clone())
        if self.mode is None or self._cur is None or self._cur != self.target:
            return self._orig(q, k, v, pe)

        n_seq, n_struct = self.n_seq, self.n_struct
        if n_seq is None or q.shape[2] != n_seq + n_struct:
            raise RuntimeError(
                "target block %s: joint sequence is %d tokens but the pre-hook saw "
                "n_seq=%s n_struct=%s" % (self.target, q.shape[2], n_seq, n_struct))

        # Same as a2holo.model.attention (RoPE, then softmax(q k^T / sqrt(d)) v) with A materialised.
        q, k = a2holo_model.apply_rope(q, k, pe)
        A = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)

        m0, m1, dev = apply_quadrant(A, n_seq, n_struct, self.mode, self.alpha)
        self.applied += 1
        self.last_target_mass = m0
        self.worst_rowsum_dev = max(self.worst_rowsum_dev, dev)
        if dev > ROWSUM_TOL:
            raise RuntimeError("rows do not sum to 1 after renormalisation "
                               "(worst deviation %.2e)" % dev)
        if self.alpha != 1.0:
            if not (m1 < m0 - 1e-12):
                raise RuntimeError(
                    "block %s mode=%s alpha=%s: targeted quadrant mass did not "
                    "drop (%.6g -> %.6g)" % (self.target, self.mode, self.alpha, m0, m1))
            self.modified += 1
        return rearrange(A @ v, "B H L D -> B L (H D)")


def verify_token_order(iv, run_forward):
    """Check that the joint token order is [seq; struct] by perturbing one stream's input to the first double
    block and measuring which half of the joint queries moves (both halves have length N, so shape cannot tell)."""
    moved = {}
    for which in ("struct", "seq"):
        iv.probe_capture, iv.probe_perturb = [], None
        run_forward()
        iv.probe_perturb = which
        run_forward()
        iv.probe_perturb, cap = None, iv.probe_capture
        iv.probe_capture = None
        if len(cap) != 2:
            raise RuntimeError("token-order probe captured %d query tensors, "
                               "expected 2" % len(cap))
        q0, q1 = cap
        d = (q1 - q0).abs().amax(dim=3).amax(dim=1).amax(dim=0)   # per-token max |dq|, [S]
        moved[which] = d
        del cap, q0, q1

    S = moved["struct"].shape[0]
    n_seq, n_struct = iv.n_seq, iv.n_struct
    if S != n_seq + n_struct:
        raise RuntimeError("probe saw S=%d but n_seq+n_struct=%d" % (S, n_seq + n_struct))
    ds, dq = moved["struct"], moved["seq"]
    report = dict(
        struct_moved_first=float(ds[:n_seq].max()), struct_moved_second=float(ds[n_seq:].min()),
        seq_moved_first=float(dq[:n_seq].min()), seq_moved_second=float(dq[n_seq:].max()))
    # |q| is O(1) after QKNorm, so moved and unmoved differ by orders of magnitude.
    ok = (report["struct_moved_first"] < 1e-6 and report["struct_moved_second"] > 1e-3
          and report["seq_moved_second"] < 1e-6 and report["seq_moved_first"] > 1e-3)
    if not ok:
        raise RuntimeError(
            "token order is not [seq; struct]: perturbing struct moved first-half queries by up to %.3e "
            "(must be 0) and second-half by at least %.3e (must be > 0); perturbing seq moved second-half "
            "by up to %.3e (must be 0) and first-half by at least %.3e (must be > 0)"
            % (report["struct_moved_first"], report["struct_moved_second"],
               report["seq_moved_second"], report["seq_moved_first"]))
    return report


def selftest():
    """Check the quadrant arithmetic and the guards without a model."""
    torch.manual_seed(0)
    n_seq = n_struct = 3
    S = n_seq + n_struct
    A0 = torch.softmax(torch.randn(1, 2, S, S), dim=-1)

    # the four slices tile the matrix exactly once
    cover = torch.zeros(S, S)
    for mode in MODES:
        for r, c in quadrant_slices(n_seq, n_struct, mode):
            cover[r, c] += 1
    assert (cover == 1).all(), "quadrant slices do not tile the matrix"

    # the two arms scale the same number of entries
    def cells(mode):
        return sum(A0[0, 0][r, c].numel() for r, c in quadrant_slices(n_seq, n_struct, mode))
    assert cells("cross") == cells("intra") == 2 * n_seq * n_struct, "arms are not size-matched"

    # alpha = 0: targeted mass goes to 0 and rows still sum to 1
    A = A0.clone()
    m0, m1, dev = apply_quadrant(A, n_seq, n_struct, "cross", 0.0)
    assert m1 < 1e-12 and dev < 1e-6, (m0, m1, dev)
    assert abs(float(A[0, 0, 0, n_seq:].sum())) < 1e-12
    assert abs(float(A[0, 0, 0, :n_seq].sum()) - 1.0) < 1e-6

    # alpha = 1 is the identity
    A = A0.clone()
    m0, m1, dev = apply_quadrant(A, n_seq, n_struct, "intra", 1.0)
    assert torch.allclose(A, A0, atol=1e-6) and abs(m1 - m0) < 1e-6

    # cross and intra mass partition the matrix
    A = A0.clone()
    mc, _, _ = apply_quadrant(A, n_seq, n_struct, "cross", 1.0)
    A = A0.clone()
    mi, _, _ = apply_quadrant(A, n_seq, n_struct, "intra", 1.0)
    assert abs(mc + mi - 1.0) < 1e-5, (mc, mi)

    # the shape guard fires on a wrong split
    try:
        apply_quadrant(A0.clone(), n_seq + 1, n_struct, "cross", 0.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("shape guard did not fire on a wrong n_seq")

    # an unknown mode is refused
    try:
        quadrant_slices(n_seq, n_struct, "both")
    except ValueError:
        pass
    else:
        raise AssertionError("mode guard did not fire")

    # struct queries (rows n_seq..) attending to seq keys (cols ..n_seq) belong to the cross arm
    def starts(mode):
        return sorted((0 if r.start is None else r.start,
                       0 if c.start is None else c.start)
                      for r, c in quadrant_slices(n_seq, n_struct, mode))
    assert starts("cross") == [(0, n_seq), (n_seq, 0)], starts("cross")
    assert starts("intra") == [(0, 0), (n_seq, n_seq)], starts("intra")

    print("[selftest] all 8 checks passed", flush=True)


def sign_p(k, n):
    """Two-sided sign test p for k successes out of n under p = 1/2."""
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=40)
    ap.add_argument("--tag", default="quadrant_intervention")
    ap.add_argument("--selftest_only", action="store_true")
    args = ap.parse_args()

    selftest()
    if args.selftest_only:
        return

    os.makedirs(os.path.dirname(os.path.join(OUT, args.tag)), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("no GPU visible -- this script needs a GPU")

    pairs = test_pairs()
    print(f"[env] device={device}  pairs={len(pairs)}  alphas={ALPHAS}  modes={MODES}  "
          f"renormalise={RENORMALISE}", flush=True)

    rows, order_reports = [], {}
    perpair = []
    for label, ckdir in MODELS:
        path = os.path.join(ckdir, "best_model.pt")
        if not os.path.exists(path):
            print(f"[skip] {label}: no checkpoint", flush=True)
            continue
        model, n_double = load_model(path, device)
        iv = QuadrantIntervenor(model)
        iv.register()
        nb = iv.n_double
        print(f"[model] {label:<18s} double_blocks={nb}", flush=True)

        base = []
        dmg = {(li, m, a): [] for li in range(nb) for m in MODES for a in ALPHAS}
        mass = {li: [] for li in range(nb)}
        ident = {(li, m): float("nan") for li in range(nb) for m in MODES}
        used, verified = 0, False

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

            def fwd():
                with torch.no_grad():
                    model(p["x"], pca, pcb, t, p["bidx"], p["seq_emb"], p["global_token"])

            def run():
                with torch.no_grad():
                    v_ca, _ = model(p["x"], pca, pcb, t, p["bidx"], p["seq_emb"], p["global_token"])
                return kabsch_rmsd((ca0 + v_ca).double(), caT.double())

            if not verified:
                iv.target, iv.mode = None, None
                rep = verify_token_order(iv, fwd)
                if iv.n_seq != N or iv.n_struct != N:
                    raise RuntimeError(
                        "expected one token per residue in each stream: N=%d but "
                        "n_seq=%d n_struct=%d" % (N, iv.n_seq, iv.n_struct))
                order_reports[label] = rep
                print(f"  [order] verified [seq ; struct], N={N}: perturbing struct "
                      f"moves only the second half (min |dq| {rep['struct_moved_second']:.2e}, "
                      f"first half exactly {rep['struct_moved_first']:.1e}); perturbing "
                      f"seq moves only the first half (min |dq| "
                      f"{rep['seq_moved_first']:.2e})", flush=True)
                verified = True

            iv.target, iv.mode, iv.alpha = None, None, 1.0
            b = run()
            base.append(b)

            for li in range(nb):
                for m in MODES:
                    for a in ALPHAS:
                        iv.target, iv.mode, iv.alpha = li, m, a
                        n0, n1 = iv.applied, iv.modified
                        dmg[(li, m, a)].append(run() - b)
                        if iv.applied != n0 + 1 or iv.modified != n1 + 1:
                            raise RuntimeError(
                                "block %d mode=%s alpha=%s: patched attention ran "
                                "%d time(s) and modified %d time(s), expected 1 and 1"
                                % (li, m, a, iv.applied - n0, iv.modified - n1))
                        if m == "cross":
                            # the intra arm's targeted share is the complement
                            mass[li].append(iv.last_target_mass)
                    if used == 0:
                        # alpha = 1 through the explicit-softmax path must reproduce the control
                        iv.target, iv.mode, iv.alpha = li, m, 1.0
                        n0 = iv.applied
                        ident[(li, m)] = abs(run() - b)
                        if iv.applied != n0 + 1:
                            raise RuntimeError("identity check did not run for "
                                               "block %d mode=%s" % (li, m))
                        if ident[(li, m)] > IDENTITY_TOL:
                            raise RuntimeError(
                                "block %d mode=%s: alpha=1.0 changed the error by "
                                "%.4f A (tol %.4f); the explicit attention does not "
                                "reproduce the fused kernel"
                                % (li, m, ident[(li, m)], IDENTITY_TOL))
            iv.target, iv.mode, iv.alpha = None, None, 1.0
            for li in range(nb):
                prow = dict(model=label, pair=pid, N=N, block=li, n_double=nb,
                            base_rmsd=b, cross_mass_frac=mass[li][-1])
                for m in MODES:
                    for a in ALPHAS:
                        prow[f"dRMSD_{m}_a{a}"] = dmg[(li, m, a)][-1]
                perpair.append(prow)
            used += 1
            del p
            gc.collect()

        for li in range(nb):
            r = dict(model=label, block=li, n_double=nb,
                     block_frac=li / max(nb - 1, 1), n=used,
                     base_rmsd=float(np.mean(base)),
                     cross_mass_frac=float(np.mean(mass[li])),
                     identity_dev=float(max(ident[(li, m)] for m in MODES)))
            for m in MODES:
                for a in ALPHAS:
                    r[f"dRMSD_{m}_a{a}"] = float(np.mean(dmg[(li, m, a)]))
            rows.append(r)

        cur = rows[-nb:]
        xc = [c["dRMSD_cross_a0.0"] for c in cur]
        xi = [c["dRMSD_intra_a0.0"] for c in cur]
        print(f"[{label}] base={np.mean(base):.3f} A | cross-off depth-mean "
              f"{np.mean(xc):+.3f} ({min(xc):+.3f}..{max(xc):+.3f}) | intra-off "
              f"{np.mean(xi):+.3f} ({min(xi):+.3f}..{max(xi):+.3f}) | ratio "
              f"{np.mean(xc) / np.mean(xi):.2f} | cross mass "
              f"{np.mean([c['cross_mass_frac'] for c in cur]):.3f} | max identity dev "
              f"{max(c['identity_dev'] for c in cur):.2e} A | worst rowsum dev "
              f"{iv.worst_rowsum_dev:.2e}", flush=True)

        iv.remove()
        del model, iv
        torch.cuda.empty_cache(); gc.collect()

    keys = (["model", "block", "n_double", "block_frac", "n", "base_rmsd",
             "cross_mass_frac", "identity_dev"]
            + [f"dRMSD_{m}_a{a}" for m in MODES for a in ALPHAS])
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

    labels = [l for l, _ in MODELS if any(r["model"] == l for r in rows)]
    print("\n[summary] depth-mean dRMSD at alpha = 0 (no block indices)", flush=True)
    print(f"{'model':<18s} {'cross-off':>10s} {'intra-off':>10s} {'ratio':>7s} "
          f"{'crossmass':>10s}", flush=True)
    wins = 0
    for l in labels:
        rr = [r for r in rows if r["model"] == l]
        c = float(np.mean([r["dRMSD_cross_a0.0"] for r in rr]))
        i = float(np.mean([r["dRMSD_intra_a0.0"] for r in rr]))
        wins += int(c > i)
        print(f"{l:<18s} {c:>10.3f} {i:>10.3f} {c / i:>7.2f} "
              f"{np.mean([r['cross_mass_frac'] for r in rr]):>10.3f}", flush=True)
    n = len(labels)
    print(f"[summary] cross-off > intra-off in {wins}/{n} trainings, two-sided "
          f"sign test p = {sign_p(max(wins, n - wins), n):.4f}", flush=True)

    print("\n[order] empirical token-order verification, all models "
          "(|dq| in the half that must NOT move / the half that MUST move)", flush=True)
    for l, rep in order_reports.items():
        print(f"  {l:<18s} perturb struct: {rep['struct_moved_first']:.2e} / "
              f"{rep['struct_moved_second']:.2e}   perturb seq: "
              f"{rep['seq_moved_second']:.2e} / {rep['seq_moved_first']:.2e}", flush=True)


if __name__ == "__main__":
    main()
