#!/usr/bin/env python3
r"""Coordinate-stream attention against the moving residues, conditional on apo geometry.

The moving-residue score of residue i is the DDM row sum y_i = sum_j |D_holo - D_apo|_ij.
The geometric rule is distance from the apo C-alpha centroid, g_i = |x_i - mean(x)|, taken
from the apo structure alone. Two conditional statistics per layer, for the received (`in`,
column mean), emitted (`out`, row mean) and fused (`fused`, row sum + column sum) attention
of the structure-to-structure quadrant:

  partial    rank-partial correlation of attention with y, holding g fixed.
  iou_resid  IoU between the top-15% by attention and the top-15% by y, computed only over
             the residues NOT in the top-15% by g. Chance is f/(2-f).

The unconditional rank correlation with y is reported as `raw`.

Reads common.test_pairs(), their embeddings and the checkpoints in common.MODELS. The input
coordinates are (1 - t) * apo + t * holo with t = common.T_VAL. Writes <tag>.tsv (per model
and layer, means over pairs) and <tag>_perpair.tsv (per model, pair and layer) under
common.OUT_DIR.

    python analysis/run.py --script analysis/attention_conditional.py --t 0.5 \
        --checkpoint seed1=checkpoints/a2holo_seed1 \
        --max_pairs 200 --tag attention_conditional
"""
import argparse, os, sys, gc
import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (          # noqa: E402
    load_model, test_pairs, load_pair, AttentionCapturer, MODELS, T_VAL, MAX_LEN, OUT_DIR)

TOP_F = 0.15                        # same top fraction as emit_recv_reference_set.py
CHANCE = TOP_F / (2 - TOP_F)


def sp(a, b):
    """Spearman without the scipy p-value machinery (called ~10^5 times)."""
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def partial(x, y, z):
    """Rank-partial correlation of x and y holding z fixed."""
    rxy, rxz, ryz = sp(x, y), sp(x, z), sp(y, z)
    den = np.sqrt(max(0.0, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return float((rxy - rxz * ryz) / den) if den > 1e-12 else np.nan


def topset(v, k):
    return set(np.argsort(-v, kind="stable")[:k].tolist())


def iou(a, b):
    return len(a & b) / len(a | b) if (a | b) else np.nan


ap = argparse.ArgumentParser()
ap.add_argument("--max_pairs", type=int, default=200)
ap.add_argument("--tag", default="attention_conditional")
a = ap.parse_args()
os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

pairs = test_pairs()
print(f"[env] device={device} candidate pairs={len(pairs)} chance={CHANCE:.4f}", flush=True)

DIRS = ("out", "in", "fused")
rows = []
# Per-pair values, recorded where each is computed (iou_resid exists only when kr > 0,
# so the aggregate lists cannot be zipped with the pair list).
perpair = []
for label, ckdir in MODELS:
    path = os.path.join(ckdir, "best_model.pt")
    if not os.path.exists(path):
        print(f"[skip] {label}", flush=True); continue
    model, nL = load_model(path, device)
    cap = AttentionCapturer(model); cap.register()
    print(f"[model] {label:<18s} depth_double={nL}", flush=True)

    acc = {f"{d}_{s}": [[] for _ in range(nL)]
           for d in DIRS for s in ("partial", "iou_resid", "raw")}
    used = 0
    for pid in pairs:
        if used >= a.max_pairs:
            break
        pair = load_pair(pid, device)
        if pair["n"] > MAX_LEN:
            continue
        data, x, N = pair["data"], pair["x"], pair["n"]
        ca0, caT, cb0, cbT = pair["ca0"], pair["caT"], pair["cb0"], pair["cbT"]

        a0 = data.pos_0.numpy().astype(np.float64)
        aT = data.pos_T.numpy().astype(np.float64)
        D0 = np.linalg.norm(a0[:, None] - a0[None], axis=-1)
        DT = np.linalg.norm(aT[:, None] - aT[None], axis=-1)
        y = np.abs(DT - D0).sum(axis=0)                 # the moving-residue score
        g = np.linalg.norm(a0 - a0.mean(0), axis=-1)    # the geometric rule, apo only
        k = max(1, int(np.ceil(N * TOP_F)))

        # residues the geometric rule does NOT flag
        resid = np.setdiff1d(np.arange(N), np.array(sorted(topset(g, k))))
        kr = max(1, int(np.ceil(len(resid) * TOP_F))) if len(resid) >= 4 else 0
        y_resid_top = topset(y[resid], kr) if kr else None

        with torch.no_grad():
            cap.clear()
            model(x, (1 - T_VAL) * ca0 + T_VAL * caT, (1 - T_VAL) * cb0 + T_VAL * cbT,
                  torch.full((1,), T_VAL, device=device),
                  pair["bidx"], pair["seq_emb"], pair["global_token"])
            cap.check()

        for li in range(nL):
            if li not in cap.captures:
                continue
            c = cap.captures[li]
            A, L = c["attn"].astype(np.float64), c["L"]
            if A.shape[0] != 2 * L:
                raise RuntimeError(f"{pid} L{li}: {A.shape} != 2L={2*L}")
            Aq = A[L:, L:]                               # coord -> coord
            prow = {"model": label, "pair": pid, "N": N, "depth": nL, "layer": li}
            for d, v in (("out", Aq.mean(axis=1)), ("in", Aq.mean(axis=0)),
                         ("fused", Aq.sum(axis=1) + Aq.sum(axis=0))):
                raw_v, par_v = sp(v, y), partial(v, y, g)
                acc[f"{d}_raw"][li].append(raw_v)
                acc[f"{d}_partial"][li].append(par_v)
                prow[f"{d}_raw"], prow[f"{d}_partial"] = raw_v, par_v
                prow[f"{d}_iou_resid"] = float("nan")
                if kr:
                    ir = iou(topset(v[resid], kr), y_resid_top)
                    acc[f"{d}_iou_resid"][li].append(ir)
                    prow[f"{d}_iou_resid"] = ir
            perpair.append(prow)
        used += 1

    for li in range(nL):
        r = {"model": label, "depth": nL, "layer": li,
             "layer_frac": li / max(1, nL - 1),
             "n": len(acc["in_raw"][li])}
        for key, vals in acc.items():
            arr = np.array([q for q in vals[li] if q == q])
            r[key] = float(arr.mean()) if arr.size else float("nan")
        rows.append(r)

    cap.remove(); del model, cap; gc.collect(); torch.cuda.empty_cache()

import csv
p = os.path.join(OUT_DIR, f"{a.tag}.tsv")
with open(p, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
print("[done] wrote", p, flush=True)
pp = os.path.join(OUT_DIR, f"{a.tag}_perpair.tsv")
with open(pp, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(perpair[0]), delimiter="\t")
    w.writeheader()
    for r in perpair:
        w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
print(f"[done] wrote {pp}  ({len(perpair)} pair x layer rows)", flush=True)
