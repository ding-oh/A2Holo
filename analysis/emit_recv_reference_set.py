"""
Emitted vs received attention per residue, for each quadrant of the joint attention map.

For every checkpoint in common.MODELS, every double-stream block and every test pair
(up to --max_pairs, length <= MAX_LEN), the head-averaged joint map A (storage order
[sequence; structure], rows normalised over all 2L keys) is split into four quadrants.
Per residue, three vectors are taken from each quadrant Aq:
    out    Aq.mean(axis=1)            emitted: row mean restricted to the quadrant
    in     Aq.mean(axis=0)            received: column mean
    fused  Aq.sum(axis=1) + Aq.sum(axis=0)
Two statistics are computed against each vector: the Spearman correlation with the
displacement |x_holo - x_apo|, and the IoU of its top 15% with the top 15% of the DDM
row sums |D_holo - D_apo|.sum(axis=0).

Because rows are normalised over all keys, the emitted vectors of the two quadrants
sharing a query row are affine complements (cp_out = 1/N - cc_out, pc_out = 1/N - pp_out),
so their rank correlations with any target have opposite signs.  Columns carry no such
constraint.

Reads common.test_pairs() and their embeddings.  Writes under common.OUT_DIR:
    <tag>.tsv          per model and layer; layer_frac = layer / (depth - 1)
    <tag>_perpair.tsv  per model, pair and layer
    <tag>_summary.tsv  depth-averaged statistics per model, with the range across models

    python analysis/run.py --script analysis/emit_recv_reference_set.py --t 0.5 \\
        --checkpoint seed1=checkpoints/a2holo_seed1 \\
        --max_pairs 150 --tag t0.5/emit_recv_reference_set
"""
import os, sys, gc, argparse
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (          # noqa: E402
    load_model, test_pairs, load_pair, AttentionCapturer, MODELS, T_VAL, MAX_LEN, OUT_DIR)

OUT = OUT_DIR

TOP_FRACTION = 0.15
CHANCE = TOP_FRACTION / (2 - TOP_FRACTION)      # = 0.0811 at f = 0.15

# (tag, slice): c is the structure stream, p the sequence (language-model) stream, query first.
# Storage order is [sequence; structure], so A[:L, :L] is p -> p.
QUADRANTS = [
    ("cc", lambda A, L: A[L:, L:]),
    ("cp", lambda A, L: A[L:, :L]),
    ("pc", lambda A, L: A[:L, L:]),
    ("pp", lambda A, L: A[:L, :L]),
]
# In each quadrant, the row mean (axis=1) is what a query residue emits (_out) and the column
# mean (axis=0) what a key residue receives (_in); _fused adds the two.


def safe_spearman(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    r, _ = spearmanr(x, y)
    return float(r)


def hotspot_iou(v, v_ddm, k):
    a = set(np.argsort(-v)[:k].tolist())
    b = set(np.argsort(-v_ddm)[:k].tolist())
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def boot_ci(vals, seed=0, n_boot=2000):
    a = np.asarray([v for v in vals if not np.isnan(v)], dtype=np.float64)
    if a.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    return float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=150)
    ap.add_argument("--tag", default="emit_recv_reference_set")
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, os.path.dirname(args.tag)), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = test_pairs()
    print(f"[env] device={device}  candidate pairs={len(pairs)}  "
          f"chance IoU={CHANCE:.4f}", flush=True)

    DIRS = ("out", "in", "fused")
    cols = [f"{q}_{d}" for q, _ in QUADRANTS for d in DIRS]
    rows = []
    # Per-pair values, recorded where each is computed: rho joins the aggregate lists only
    # when defined, so those lists can be shorter than the pair list.
    perpair = []

    for label, ckdir in MODELS:
        path = os.path.join(ckdir, "best_model.pt")
        if not os.path.exists(path):
            print(f"[skip] {label}: no checkpoint at {path}", flush=True)
            continue
        model, n_layers = load_model(path, device)
        torch.cuda.empty_cache()
        cap = AttentionCapturer(model)
        cap.register()
        print(f"[model] {label:<18s} depth_double={n_layers}", flush=True)

        rho = {c: [[] for _ in range(n_layers)] for c in cols}
        iou = {c: [[] for _ in range(n_layers)] for c in cols}
        used = 0
        for pid in pairs:
            if used >= args.max_pairs:
                break
            pair = load_pair(pid, device)
            if pair["n"] > MAX_LEN:
                continue
            data, x, bidx, N = pair["data"], pair["x"], pair["bidx"], pair["n"]
            ca0, caT, cb0, cbT = pair["ca0"], pair["caT"], pair["cb0"], pair["cbT"]

            a0 = data.pos_0.numpy().astype(np.float64)
            aT = data.pos_T.numpy().astype(np.float64)
            disp = np.linalg.norm(aT - a0, axis=-1)          # |dx|, rotation invariant
            D0 = np.linalg.norm(a0[:, None] - a0[None], axis=-1)
            DT = np.linalg.norm(aT[:, None] - aT[None], axis=-1)
            v_ddm = np.abs(DT - D0).sum(axis=0)
            k = max(1, int(np.ceil(N * TOP_FRACTION)))

            pos_ca = (1 - T_VAL) * ca0 + T_VAL * caT
            pos_cb = (1 - T_VAL) * cb0 + T_VAL * cbT
            t = torch.full((1,), T_VAL, device=device)
            cap.clear()
            with torch.no_grad():
                model(x, pos_ca, pos_cb, t, bidx, pair["seq_emb"], pair["global_token"])
            cap.check()

            for li in range(n_layers):
                if li not in cap.captures:
                    continue
                c = cap.captures[li]
                A, L = c["attn"].astype(np.float64), c["L"]
                if A.shape[0] != 2 * L:
                    raise RuntimeError(f"{pid} L{li}: attn {A.shape} != 2L={2*L}")
                for tag, sl in QUADRANTS:
                    Aq = sl(A, L)
                    # "fused" adds the row and the column sums.  Sum and mean differ
                    # by the constant 1/L, which cannot change a top-k ranking, so
                    # out/in stay comparable to it.
                    for d, v in (("out",   Aq.mean(axis=1)),
                                 ("in",    Aq.mean(axis=0)),
                                 ("fused", Aq.sum(axis=1) + Aq.sum(axis=0))):
                        key = f"{tag}_{d}"
                        r = safe_spearman(v, disp)
                        if not np.isnan(r):
                            rho[key][li].append(r)
                        iou_val = hotspot_iou(v, v_ddm, k)
                        iou[key][li].append(iou_val)
                        if not (perpair and perpair[-1]["model"] == label
                                and perpair[-1]["pair"] == pid and perpair[-1]["layer"] == li):
                            perpair.append(dict(model=label, pair=pid, N=N,
                                                depth=n_layers, layer=li))
                        perpair[-1][f"rho_{key}"] = r
                        perpair[-1][f"iou_{key}"] = iou_val
            used += 1
            del pair, data, a0, aT, D0, DT
            gc.collect()

        for li in range(n_layers):
            r = dict(model=label, depth=n_layers, layer=li,
                     layer_frac=li / max(n_layers - 1, 1), n=used)
            for c in cols:
                r[f"rho_{c}"] = float(np.mean(rho[c][li])) if rho[c][li] else np.nan
                m, lo, hi = boot_ci(iou[c][li], seed=li)
                r[f"iou_{c}"], r[f"iou_{c}_lo"], r[f"iou_{c}_hi"] = m, lo, hi
            rows.append(r)

        cur = rows[-n_layers:]
        for c in ("cc_out", "cc_in", "cc_fused", "cp_out", "pc_in"):
            v = [cur[i][f"rho_{c}"] for i in range(n_layers)]
            iv = [cur[i][f"iou_{c}"] for i in range(n_layers)]
            print(f"[{label}] {c:<7s} rho depth-mean={np.nanmean(v):+.4f} "
                  f"|max|={np.nanmax(np.abs(v)):.4f}   "
                  f"IoU depth-mean={np.nanmean(iv):.4f} "
                  f"({np.nanmean(iv)/CHANCE:.2f}x chance)", flush=True)
        del model, cap
        torch.cuda.empty_cache()
        gc.collect()

    keys = ["model", "depth", "layer", "layer_frac", "n"]
    for c in cols:
        keys += [f"rho_{c}", f"iou_{c}", f"iou_{c}_lo", f"iou_{c}_hi"]
    tsv = os.path.join(OUT, f"{args.tag}.tsv")
    with open(tsv, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[k]:.5f}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")
    print(f"[done] wrote {tsv}", flush=True)
    pp = os.path.join(OUT, f"{args.tag}_perpair.tsv")
    pkeys = list(perpair[0]) if perpair else []
    with open(pp, "w") as f:
        f.write("\t".join(pkeys) + "\n")
        for r in perpair:
            f.write("\t".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                              for k in pkeys) + "\n")
    print(f"[done] wrote {pp}  ({len(perpair)} pair x layer rows)", flush=True)

    # Depth-averaged summary; no layer index appears in it.
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    summ = os.path.join(OUT, f"{args.tag}_summary.tsv")
    with open(summ, "w") as f:
        f.write("quantity\tstat\t" + "\t".join(models) + "\tmin\tmax\tn_models\n")
        for c in cols:
            for stat, pick in (("rho_depthmean", lambda rs, c=c: np.nanmean([r[f"rho_{c}"] for r in rs])),
                               ("iou_depthmean", lambda rs, c=c: np.nanmean([r[f"iou_{c}"] for r in rs])),
                               ("iou_x_chance",  lambda rs, c=c: np.nanmean([r[f"iou_{c}"] for r in rs]) / CHANCE)):
                vals = [pick([r for r in rows if r["model"] == m]) for m in models]
                f.write(f"{c}\t{stat}\t" + "\t".join(f"{v:.5f}" for v in vals)
                        + f"\t{np.nanmin(vals):.5f}\t{np.nanmax(vals):.5f}\t{len(vals)}\n")
    print(f"[done] wrote {summ}", flush=True)
    print(f"[note] chance IoU = {CHANCE:.5f}; report ranges across models, not layer indices.",
          flush=True)


if __name__ == "__main__":
    main()
