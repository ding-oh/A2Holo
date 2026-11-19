#!/usr/bin/env python3
r"""Reduce a per-layer TSV written by an analysis script to per-model depth means.

Each statistic is averaged over depth within a model and reported per model, with the
range across models and the model count.  An ordering between two statistics is
counted over models and given a two-sided sign test.  No layer index is returned.
Reads one TSV and prints to stdout; KIND names the script that wrote it:

  emitrecv  emit_recv_reference_set.py, emit_recv_controls.py
  alpha     alpha_scaling_reference_set.py
  cond      attention_conditional.py
  quadrant  quadrant_intervention.py

  python analysis/summarize.py KIND TSV
  python analysis/summarize.py emitrecv outputs/analysis/emit_recv_random.tsv
"""
import csv, sys
from collections import defaultdict
from math import comb

CHANCE = 0.15 / (2 - 0.15)


def sign_p(k, n):
    if n == 0:
        return float("nan")
    return min(1.0, sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n * 2)


def load(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def grp(rows, key="model"):
    d = defaultdict(list)
    for r in rows:
        d[r[key]].append(r)
    return d


def num(r, c):
    v = r.get(c, "")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def depthmean(rs, c):
    v = [x for x in (num(r, c) for r in rs) if x is not None]
    return sum(v) / len(v) if v else None


def report(title, per_model, fmt="{:+.3f}", scale=1.0, better=lambda a, b: a > b):
    print(f"\n== {title}")
    vals = {m: v * scale for m, v in per_model.items() if v is not None}
    for m, v in vals.items():
        print(f"   {m:<20} {fmt.format(v)}")
    if vals:
        lo, hi = min(vals.values()), max(vals.values())
        print(f"   {'RANGE':<20} {fmt.format(lo)} .. {fmt.format(hi)}   (n={len(vals)} models)")
    return vals


def cmp_report(title, a, b, la, lb):
    """Ordering-as-event: how many models put a above b, with a sign test."""
    ks = [m for m in a if m in b and a[m] is not None and b[m] is not None]
    k = sum(1 for m in ks if a[m] > b[m])
    print(f"\n== {title}")
    print(f"   {la} > {lb} in {k}/{len(ks)} models, two-sided sign test p = {sign_p(max(k, len(ks)-k), len(ks)):.4f}")
    rat = [a[m] / b[m] for m in ks if b[m] not in (None, 0)]
    if rat:
        print(f"   ratio range {min(rat):.2f} .. {max(rat):.2f}")


def do_emitrecv(rows):
    g = grp(rows)
    out = {m: depthmean(rs, "iou_cc_out") for m, rs in g.items()}
    inn = {m: depthmean(rs, "iou_cc_in") for m, rs in g.items()}
    fus = {m: depthmean(rs, "iou_cc_fused") for m, rs in g.items()}
    a = report("received, depth-mean IoU (x chance)", inn, "{:.2f}", 1 / CHANCE)
    b = report("emitted,  depth-mean IoU (x chance)", out, "{:.2f}", 1 / CHANCE)
    report("fused,    depth-mean IoU (x chance)", fus, "{:.2f}", 1 / CHANCE)
    cmp_report("ASYMMETRY", a, b, "received", "emitted")


def do_alpha(rows):
    g = grp(rows)
    db = {m: depthmean([r for r in rs if r["block_type"] == "double"], "dRMSD_a0.0") for m, rs in g.items()}
    sg = {m: depthmean([r for r in rs if r["block_type"] == "single"], "dRMSD_a0.0") for m, rs in g.items()}
    a = report("double-stream blocks, mean dRMSD (A)", db)
    b = report("single-stream blocks, mean dRMSD (A)", sg)
    cmp_report("BLOCK TYPE", a, b, "double", "single")
    report("control error (base RMSD, A) -- NOT an accuracy number",
           {m: num(rs[0], "base_rmsd") for m, rs in g.items()}, "{:.3f}")


def do_cond(rows):
    g = grp(rows)
    for stat, lab in (("partial", "partial rank corr. with displacement, holding apo geometry fixed"),
                      ("iou_resid", "IoU inside the set the geometric rule misses (x chance)"),
                      ("raw", "raw rank corr. with displacement (no control)")):
        sc = 1 / CHANCE if stat == "iou_resid" else 1.0
        f = "{:.2f}" if stat == "iou_resid" else "{:+.3f}"
        a = report(f"received -- {lab}", {m: depthmean(rs, f"in_{stat}") for m, rs in g.items()}, f, sc)
        b = report(f"emitted  -- {lab}", {m: depthmean(rs, f"out_{stat}") for m, rs in g.items()}, f, sc)
        cmp_report(f"ASYMMETRY on {stat}", a, b, "received", "emitted")
        pos = sum(1 for v in a.values() if v > (1.0 if stat == "iou_resid" else 0.0))
        print(f"   received above null in {pos}/{len(a)} models, p = {sign_p(max(pos, len(a)-pos), len(a)):.4f}")


def do_quadrant(rows):
    """Columns: model block n_double block_frac n base_rmsd cross_mass_frac
    identity_dev dRMSD_{cross,intra}_a{0.0,0.5}.

      * dRMSD_* is already a delta against that model's own control on the same pair.
        Never subtract base_rmsd again.  Positive = the intervention made it worse;
        negative values are real, not errors to clip.
      * identity_dev prints as 0.000000 for every row: the TSV is written with %.6f and
        the values are of order 1e-7, so the zero is rounding.
      * this TSV holds double blocks only, so block_frac is normalised over the double
        stack alone and is not comparable with alpha_scaling's, which spans all blocks.
    """
    hdr = rows[0]
    for c in ("dRMSD_cross_a0.0", "dRMSD_intra_a0.0"):
        if c not in hdr:
            sys.exit("expected column %r missing; header is %s" % (c, list(hdr)))
    g = grp(rows)
    cr = {m: depthmean(rs, "dRMSD_cross_a0.0") for m, rs in g.items()}
    it = {m: depthmean(rs, "dRMSD_intra_a0.0") for m, rs in g.items()}
    a = report("cross-stream quadrants off (a=0), mean dRMSD (A)", cr)
    b = report("intra-stream quadrants off (a=0), mean dRMSD (A)", it)
    cmp_report("CROSS vs INTRA", a, b, "cross", "intra")
    report("cross share of attention mass (0.5 = uniform)",
           {m: depthmean(rs, "cross_mass_frac") for m, rs in g.items()}, "{:.3f}")
    d5 = {m: depthmean(rs, "dRMSD_cross_a0.5") for m, rs in g.items()}
    report("cross arm at a=0.5, mean dRMSD (A)", d5)
    print("\n   a=0.5 is NOT half the damage: rows are renormalised, so the surviving")
    print("   cross mass is 0.5m/(1-0.5m) and the dose axis is not linear in dRMSD.")
    print("\n   NOTE: intra-off also removes the self-attention diagonal and cross-off does")
    print("   not, which tilts the comparison TOWARD intra.  A cross-wins result is")
    print("   therefore conservative; an intra-wins result is confounded and licenses")
    print("   nothing.  identity_dev in the TSV is rounded to 0.000000 -- read the")
    print("   stdout of quadrant_intervention.py for the real 1e-07 values.")


KIND = {"emitrecv": do_emitrecv, "alpha": do_alpha, "cond": do_cond, "quadrant": do_quadrant}
if len(sys.argv) != 3 or sys.argv[1] not in KIND:
    sys.exit(__doc__)
rs = load(sys.argv[2])
print(f"{sys.argv[2]}  rows={len(rs)}  models={len(grp(rs))}  n={sorted({r.get('n','-') for r in rs})}")
KIND[sys.argv[1]](rs)
