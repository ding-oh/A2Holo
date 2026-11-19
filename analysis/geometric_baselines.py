"""Apo-only geometric baselines for the moving-residue overlap.

How well rules that use only the apo structure (no model, no holo, no t) recover the moving
residues, on the same test pairs and with the same ground truth as the attention analyses, so
attention overlaps can be read against these rules as well as against the combinatorial chance
level f/(2-f) = 0.0811. Ground truth, pair list, MAX_LEN filter and boot_ci(n_boot=2000) are
those of emit_recv_reference_set.py: v_ddm = |D_T - D_0|.sum(axis=0), the top k = ceil(0.15 N),
IoU = |a & b| / |a | b|.

  dist_centroid  top-15% by |CA_i - centroid(CA)|          (periphery)
  contact_low    top-15% by fewest neighbours within 10 A  (least buried)
  terminal       top-15% by |i - N/2|                      (chain ends)
  random         a uniform random 15% subset               (checks the construction: it has to
                                                            land on the chance level)

Contact counts are integers and |i - N/2| is symmetric, so both tie heavily at the top-k
boundary, where argsort would prefer the lower residue index -- for `terminal`, systematically
the N-terminus. Those two break ties uniformly at random (fixed seed); the float scores use
plain argsort(-v). The TSV carries every baseline's boundary-tie count.

Reads the test pairs from common.DATA_DIR; no checkpoint. Writes <tag>.tsv (per pair) and
<tag>_summary.tsv (per baseline, means over pairs with a bootstrap CI) under common.OUT_DIR.
run.py requires --t, which this script does not use.

    python analysis/run.py --script analysis/geometric_baselines.py --t 0 \\
        --max_pairs 150 --tag geometric_baselines
"""
import os, sys, gc, argparse, time
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, MAX_LEN, OUT_DIR, SPLITS, test_pairs  # noqa: E402
from a2holo.data import split_ids  # noqa: E402

TOP_FRACTION = 0.15                 # same top fraction as emit_recv_reference_set.py
CHANCE = TOP_FRACTION / (2 - TOP_FRACTION)      # = 0.0811 at f = 0.15
CONTACT_CUTOFF = 10.0               # Angstrom, CA-CA

BASELINES = ["dist_centroid", "contact_low", "terminal", "random"]
TIED = ["contact_low", "terminal"]      # integer / symmetric scores


def topk_set(v, k, tiebreak=None):
    """Indices of the k largest entries of v.

    tiebreak=None is np.argsort(-v)[:k], ties resolved by index, as in
    emit_recv_reference_set.py.  Passing a random vector resolves ties uniformly
    instead -- needed for the integer-valued scores.
    """
    if tiebreak is None:
        return set(np.argsort(-v)[:k].tolist())
    return set(np.lexsort((tiebreak, -v))[:k].tolist())


def boundary_ties(v, k):
    """How many entries share the value of the k-th largest, i.e. how much of
    the top-k set was decided by the tie-break rather than by the score."""
    if k >= v.size:
        return 0
    thr = np.sort(v)[::-1][k - 1]
    return int((v == thr).sum())


def safe_spearman(x, y):
    """Spearman correlation, NaN when either input is constant (as in emit_recv_reference_set.py)."""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    r, _ = spearmanr(x, y)
    return float(r)


def iou(a, b):
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def boot_ci(vals, seed=0, n_boot=2000):
    """Mean and 95% percentile bootstrap CI over the non-NaN values (as in emit_recv_reference_set.py)."""
    a = np.asarray([v for v in vals if not np.isnan(v)], dtype=np.float64)
    if a.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    return float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=150)
    ap.add_argument("--tag", default="geometric_baselines")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rand_reps", type=int, default=200,
                    help="extra random draws per pair, averaged, as a "
                         "low-variance check that chance really is 0.0811")
    ap.add_argument("--tb_reps", type=int, default=20,
                    help="tie-break redraws for the integer-valued scores, to "
                         "show how much of the result the tie-break decides")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    pairs = test_pairs()
    print(f"[env] candidate pairs={len(pairs)} of {len(split_ids(SPLITS, 'test'))} test  "
          f"chance IoU={CHANCE:.4f}  cutoff={CONTACT_CUTOFF} A", flush=True)

    rows = []
    rand_rep_means = []
    used = 0
    for pi, pid in enumerate(pairs):
        if used >= args.max_pairs:
            break
        pt_path = os.path.join(DATA_DIR, f"{pid}.pt")
        pt_mtime = os.stat(pt_path).st_mtime
        data = torch.load(pt_path, weights_only=False)
        if data.x.shape[0] > MAX_LEN:
            del data
            continue
        N = int(data.x.shape[0])
        a0 = data.pos_0.numpy().astype(np.float64)
        aT = data.pos_T.numpy().astype(np.float64)
        if a0.shape[0] != N or aT.shape[0] != N:
            raise RuntimeError(f"{pid}: pos shape {a0.shape}/{aT.shape} vs N={N}")

        # Ground truth, as in emit_recv_reference_set.py.
        D0 = np.linalg.norm(a0[:, None] - a0[None], axis=-1)
        DT = np.linalg.norm(aT[:, None] - aT[None], axis=-1)
        v_ddm = np.abs(DT - D0).sum(axis=0)
        k = max(1, int(np.ceil(N * TOP_FRACTION)))
        gt = topk_set(v_ddm, k)
        if len(gt) != k:
            raise RuntimeError(f"{pid}: |gt|={len(gt)} != k={k}")

        rng = np.random.default_rng(args.seed * 1_000_003 + pi)

        # Baselines, from the apo structure alone.
        v_cen = np.linalg.norm(a0 - a0.mean(axis=0, keepdims=True), axis=-1)
        n_contact = (D0 < CONTACT_CUTOFF).sum(axis=1) - 1      # exclude self
        v_con = -n_contact.astype(np.float64)                  # fewest = highest
        idx = np.arange(N, dtype=np.float64)
        v_term = np.abs(idx - N / 2.0)

        sets = {
            "dist_centroid": topk_set(v_cen, k),
            "contact_low":   topk_set(v_con, k, rng.random(N)),
            "terminal":      topk_set(v_term, k, rng.random(N)),
            "random":        set(rng.permutation(N)[:k].tolist()),
        }
        ties = {
            "dist_centroid": boundary_ties(v_cen, k),
            "contact_low":   boundary_ties(v_con, k),
            "terminal":      boundary_ties(v_term, k),
            "random":        0,
        }
        for b, s in sets.items():
            if len(s) != k:
                raise RuntimeError(f"{pid}/{b}: |set|={len(s)} != k={k}")

        # Low-variance chance estimate for this pair; not used in the CI.
        rr = [iou(gt, set(rng.permutation(N)[:k].tolist()))
              for _ in range(args.rand_reps)]
        rand_rep_means.append(float(np.mean(rr)))

        # How much of each tied baseline the tie-break decides rather than the score.
        scores = {"dist_centroid": v_cen, "contact_low": v_con,
                  "terminal": v_term, "random": None}
        tb = {}
        for b in TIED:
            vv = [iou(gt, topk_set(scores[b], k, rng.random(N)))
                  for _ in range(args.tb_reps)]
            tb[b] = (float(np.mean(vv)), float(np.min(vv)), float(np.max(vv)))

        r = dict(pid=pid, N=N, k=k, pt_mtime=pt_mtime, f=k / N, chance_pair=(k / N) / (2 - k / N),
                 ddm_ties=boundary_ties(v_ddm, k),
                 rand_rep=rand_rep_means[-1])
        for b in BASELINES:
            r[f"iou_{b}"] = iou(gt, sets[b])
            r[f"ties_{b}"] = ties[b]
            # Rank agreement over all residues; the top-k IoU is one threshold on it.
            r[f"rho_{b}"] = (safe_spearman(scores[b], v_ddm)
                             if scores[b] is not None else np.nan)
        for b in TIED:
            r[f"iou_{b}_tbmean"], r[f"iou_{b}_tblo"], r[f"iou_{b}_tbhi"] = tb[b]
        # The ground truth against itself must be exactly 1.
        r["iou_self"] = iou(gt, gt)
        rows.append(r)
        used += 1
        del data, a0, aT, D0, DT
        gc.collect()

    if not rows:
        raise RuntimeError("no pairs passed the filters")

    keys = ["pid", "N", "k", "pt_mtime", "f", "chance_pair", "ddm_ties", "rand_rep", "iou_self"]
    for b in BASELINES:
        keys += [f"iou_{b}", f"ties_{b}", f"rho_{b}"]
    for b in TIED:
        keys += [f"iou_{b}_tbmean", f"iou_{b}_tblo", f"iou_{b}_tbhi"]
    tsv = os.path.join(OUT_DIR, f"{args.tag}.tsv")
    with open(tsv, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(f"{r[kk]:.5f}" if isinstance(r[kk], float) else str(r[kk])
                               for kk in keys) + "\n")
    print(f"[done] wrote {tsv}  (n={len(rows)} pairs)", flush=True)

    summ = os.path.join(OUT_DIR, f"{args.tag}_summary.tsv")
    lines = []
    for bi, b in enumerate(BASELINES):
        m, lo, hi = boot_ci([r[f"iou_{b}"] for r in rows],
                            seed=bi, n_boot=args.n_boot)
        tbm = (float(np.mean([r[f"iou_{b}_tbmean"] for r in rows]))
               if b in TIED else m)
        rv = [r[f"rho_{b}"] for r in rows if not np.isnan(r[f"rho_{b}"])]
        rho = float(np.mean(rv)) if rv else float("nan")
        lines.append((b, len(rows), m, lo, hi, m / CHANCE, lo / CHANCE,
                      hi / CHANCE, tbm, tbm / CHANCE, rho))
    with open(summ, "w") as fh:
        fh.write("baseline\tn_pairs\tmean_iou\tci_lo\tci_hi\t"
                 "x_chance\tx_chance_lo\tx_chance_hi\t"
                 "mean_iou_tiebreak_avg\tx_chance_tiebreak_avg\trho_vs_ddm\n")
        for t in lines:
            fh.write(f"{t[0]}\t{t[1]}\t" + "\t".join(f"{v:.5f}" for v in t[2:]) + "\n")
    print(f"[done] wrote {summ}", flush=True)

    print(f"[chance] combinatorial f/(2-f) at f=0.15      = {CHANCE:.5f}")
    print(f"[chance] mean per-pair (k/N)/(2-k/N), ceil'd  = "
          f"{np.mean([r['chance_pair'] for r in rows]):.5f}")
    print(f"[chance] empirical, {args.rand_reps} draws x {len(rows)} pairs  = "
          f"{np.mean(rand_rep_means):.5f}", flush=True)
    print(f"[sanity] iou(gt,gt) min={min(r['iou_self'] for r in rows):.5f} "
          f"max={max(r['iou_self'] for r in rows):.5f}  (must be 1)")
    mt = [r["pt_mtime"] for r in rows]
    print(f"[input] pair .pt mtimes span "
          f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(min(mt)))} .. "
          f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(max(mt)))}", flush=True)
    print(f"[sanity] N range {min(r['N'] for r in rows)}-{max(r['N'] for r in rows)}  "
          f"k range {min(r['k'] for r in rows)}-{max(r['k'] for r in rows)}  "
          f"DDM boundary ties max={max(r['ddm_ties'] for r in rows)}")
    for b in BASELINES:
        tmax = max(r[f"ties_{b}"] for r in rows)
        tmean = np.mean([r[f"ties_{b}"] for r in rows])
        print(f"[ties] {b:<14s} boundary ties mean={tmean:.2f} max={tmax}")
    print()
    print(f"{'baseline':<14s} {'mean IoU':>9s} {'95% CI':>19s} {'x chance':>9s}"
          f" {'tb-avg':>8s} {'rho_ddm':>8s}")
    for b, n, m, lo, hi, xm, xlo, xhi, tbm, xtb, rho in lines:
        print(f"{b:<14s} {m:9.4f} [{lo:7.4f},{hi:7.4f}] {xm:8.2f}x  "
              f"[{xlo:.2f},{xhi:.2f}] {xtb:7.2f}x {rho:+8.3f}", flush=True)
    print(f"\n[note] chance IoU = {CHANCE:.5f}; n = {len(rows)} pairs; "
          f"no layer index appears in this analysis.", flush=True)


if __name__ == "__main__":
    main()
