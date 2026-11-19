"""Comparison tables: A2Holo models against sampled baselines, scored from PDB files.

Every structure is scored per test pair of --splits:
  accuracy      Cα RMSD to holo after Kabsch superposition, TM-score (USalign, normalised by the
                holo length), Cα lDDT
  displacement  Cα RMSD to apo after Kabsch superposition
  geometry      Cα–Cα bond violations |d − 3.8| > 0.5 Å and > 0.2 Å (rates over valid bonds), Cα
                clashes (d < 3.4 Å, |i − j| >= 3, all residues), (θ, τ) outliers, Rg − Rg(apo), and,
                where Cβ is present, Cα–Cβ violations |d − 1.532| > 0.05 Å and Cβ clashes
                (d < 3.4 Å, |i − j| >= 2)

A bond (i, i+1) is valid when its Cα–Cα distance is below 4.5 Å in both apo and holo, so missing
residues never count as violations; a pair with no valid bond has no bond or (θ, τ) values. A (θ, τ)
window is 4 consecutive Cα joined by valid bonds, θ the angle at the second and τ the dihedral, binned
in 5° x 10° cells. Over the windows of the train holo structures (--data_dir), the outlier cells are
those with count <= t, t the largest count for which these cells, empty ones included, hold at most
0.5 % of the windows. Without --data_dir the (θ, τ) column is left out.

Backbone section (the D3PM measures of Appendix A.8; each baseline keeps its TM-best sample): Cα–Cα
violation rate over the bonds holo keeps (d <= 4.5 Å) and their largest gap; chain breaks (Cα–Cα
> 4.0 Å within apo segments) introduced over apo, with an exact two-sided sign test per pair; TM-score
gain over apo. For D3PM, pass --splits data/d3pm_splits.json.

A baseline gives several samples per pair, and each rule keeps one:
  sample0   the first sample
  geometry  geometry-best: fewest Cα–Cα violations at 0.5 Å, then fewest Cα clashes, then the lower
            sample index; it needs no holo structure and is the main rule
  oracle    RMSD oracle: lowest Cα RMSD to holo, which needs the answer

The first --model is the main A2Holo row. Statistics are over pairs; Δ are per pair, counted as
better/worse with ties left out, with a two-sided Wilcoxon signed-rank test. The main model is compared
with each baseline under each rule and, on TM-score, with each baseline's TM-best sample (an oracle).
The displacement-matched comparison takes the main model
and baseline sample 0; in each pair, whichever moved further from apo is moved back along
apo + α (superposed − apo) to the other's displacement before its Cα–Cα violations are counted.

    python analysis/splitA_tables.py --splits data/apollodiff_splitA.json --ref_dir eval/recipe \\
        --model recipe=eval/recipe/pred,eval/recipe.json --model depth4=eval/depth4/pred,eval/depth4.json \\
        --model no_plm=eval/no_plm/pred,eval/no_plm.json \\
        --model seq_from_structure=eval/seq_from_structure/pred,eval/seq_from_structure.json \\
        --checkpoint recipe=checkpoints/splitA_recipe/best_model.pt \\
        --baseline SBAlign=baselines/sbalign --baseline ABM=baselines/abm --baseline FDBM=baselines/fdbm \\
        --size depth4=recipe --size no_plm=seq_from_structure \\
        --interaction recipe,depth4,seq_from_structure,no_plm --displacement recipe=seq_from_structure \\
        --data_dir data/d3pm/pairs --usalign USalign --out_json outputs/splitA_tables.json

--ref_dir holds target/ and apo/ as evaluate.py --pdb_dir writes them. A model is a directory of
<pair>.pdb (evaluate.py's pred/), optionally with its evaluate.py results JSON for the Cβ RMSD, and
optionally a checkpoint (--checkpoint) for its blocks, parameters and selected epoch. A baseline is a
directory of sample<k>/<pair>.pdb. Per pair: --size S=L, L − S; --interaction AL,AS,BL,BS,
(AL − AS) − (BL − BS); --displacement A=B, displacement from apo, A − B.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, kabsch_rmsd, kabsch_rotation  # noqa: E402
from a2holo.geometry import ca_lddt  # noqa: E402
from a2holo.model import A2Holo  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tm_score import score_pair  # noqa: E402

BOND, BOND_TOL, PV_TOL = 3.8, 0.5, 0.2      # Cα–Cα ideal and the two violation thresholds, Å
GAP = 4.5                                   # apo or holo Cα–Cα at or above this spans missing residues
CLASH, CLASH_SEP, CB_CLASH_SEP = 3.4, 3, 2
CACB, CACB_TOL = 1.532, 0.05
BREAK = 4.0                                 # a chain break: consecutive Cα farther apart than this, Å
TH_BIN, TA_BIN, OUTLIER_MASS = 5.0, 10.0, 0.005
NTH, NTA = int(180 / TH_BIN), int(360 / TA_BIN)
RULES = ("sample0", "geometry", "oracle")
RULE_LABEL = dict(sample0="sample 0", geometry="geometry-best", oracle="RMSD oracle")
NAN = float("nan")


def read_pdb(path):
    """Cα [N, 3] and Cβ [N, 3] (NaN where a residue has none), first model, in file order."""
    res = {}
    with open(path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            name = line[12:16].strip()
            if line.startswith(("ATOM", "HETATM")) and name in ("CA", "CB"):
                res.setdefault(line[21:27], {}).setdefault(name, [float(line[c:c + 8]) for c in (30, 38, 46)])
    ca = np.array([r["CA"] for r in res.values()], dtype=np.float64)
    cb = np.array([r.get("CB", [NAN] * 3) for r in res.values()], dtype=np.float64)
    return ca, cb


def rmsd(p, q):
    return kabsch_rmsd(torch.from_numpy(p), torch.from_numpy(q))


def superpose(p, q):
    """p superposed on q."""
    P, Q = torch.from_numpy(p), torch.from_numpy(q)
    return ((P - P.mean(0)) @ kabsch_rotation(P, Q) + Q.mean(0)).numpy()


def bond_lengths(x):
    return np.linalg.norm(x[1:] - x[:-1], axis=1)


def pair_count(x, sep, thresh, mask=None):
    i, j = np.triu_indices(len(x), sep)
    if mask is not None:
        keep = mask[i] & mask[j]
        i, j = i[keep], j[keep]
    return int((np.linalg.norm(x[i] - x[j], axis=1) < thresh).sum())


def radius_of_gyration(x):
    return float(np.sqrt(((x - x.mean(0)) ** 2).sum(-1).mean()))


def theta_tau(x, windows):
    """θ at the second Cα and τ over the four, for the windows (k..k+3) set in `windows`."""
    u, v = x[:-2] - x[1:-1], x[2:] - x[1:-1]
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-8)
    theta = np.arccos(np.clip(cos, -1 + 1e-7, 1 - 1e-7)) * (180.0 / np.pi)
    b1, b2, b3 = x[1:-2] - x[:-3], x[2:-1] - x[1:-2], x[3:] - x[2:-1]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8))
    tau = np.arctan2((m1 * n2).sum(-1), (n1 * n2).sum(-1)) * (180.0 / np.pi)
    return theta[:len(windows)][windows], tau[windows]


def cells(theta, tau):
    return (np.clip(np.floor(theta / TH_BIN).astype(int), 0, NTH - 1),
            np.clip(np.floor((tau + 180.0) / TA_BIN).astype(int), 0, NTA - 1))


def masks(apo, holo):
    valid = (bond_lengths(apo) < GAP) & (bond_lengths(holo) < GAP)
    return valid, valid[:-2] & valid[1:-1] & valid[2:]


def outlier_cells(data_dir, train_ids):
    """The rarest (θ, τ) cells of the train holo structures, and a summary of the reference."""
    hist = np.zeros((NTH, NTA), dtype=np.int64)
    for pid in train_ids:
        d = torch.load(os.path.join(data_dir, f"{pid}.pt"), weights_only=False)
        apo, holo = d.pos_0.double().numpy(), d.pos_T.double().numpy()
        np.add.at(hist, cells(*theta_tau(holo, masks(apo, holo)[1])), 1)
    thresh = -1
    for c in np.unique(hist):
        if hist[hist <= c].sum() > OUTLIER_MASS * hist.sum():
            break
        thresh = int(c)
    out = hist <= thresh
    return out, dict(train_pairs=len(train_ids), windows=int(hist.sum()), count_threshold=thresh,
                     outlier_cells=int(out.sum()), cells=int(hist.size), outlier_mass=float(hist[out].sum() / hist.sum()))


def geometry(x, cb, ref, outliers):
    """Bond, clash, (θ, τ), Rg and displacement values of one structure of a pair."""
    rate = lambda n, m: n / m if m else NAN
    dev = bond_lengths(x)[ref["valid"]] - BOND
    nb, n05, n02 = dev.size, int((np.abs(dev) > BOND_TOL).sum()), int((np.abs(dev) > PV_TOL).sum())
    g = dict(n_bonds=nb, n05=n05, v05=rate(n05, nb), v02=rate(n02, nb), clash=pair_count(x, CLASH_SEP, CLASH),
             disp=rmsd(x, ref["apo"]), drg=radius_of_gyration(x) - ref["rg_apo"])
    d = bond_lengths(x)
    kept = d[ref["holo_keep"]]
    g.update(breaks=int(((d > BREAK) & ref["apo_seg"]).sum()),
             v05_holo=rate(int((np.abs(kept - BOND) > BOND_TOL).sum()), kept.size),
             max_gap=float(kept.max()) if kept.size else NAN)
    if outliers is not None:
        theta, tau = theta_tau(x, ref["windows"])
        g.update(n_windows=len(theta), n_out=int(outliers[cells(theta, tau)].sum()))
        g["theta_tau_out"] = rate(g["n_out"], g["n_windows"])
    m = ~np.isnan(cb[:, 0])
    if m.any():
        dd = np.linalg.norm(cb[m] - x[m], axis=1)
        g["cacb"] = rate(int((np.abs(dd - CACB) > CACB_TOL).sum()), dd.size) if nb else NAN
        g["cbclash"] = pair_count(cb, CB_CLASH_SEP, CLASH, m)
    return g


def v05_after_shrink(x, ref, target):
    """Cα–Cα violation rate of x moved toward apo to a Kabsch RMSD to apo of `target` (bisection)."""
    apo, al = ref["apo"], superpose(x, ref["apo"])
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rmsd(apo + mid * (al - apo), apo) < target else (lo, mid)
    dev = bond_lengths(apo + 0.5 * (lo + hi) * (al - apo))[ref["valid"]] - BOND
    return int((np.abs(dev) > BOND_TOL).sum()) / dev.size if dev.size else NAN


def tm_scores(jobs, usalign, cache_path, workers):
    """tm_target of each (pred, target) PDB pair, cached by file content."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    digest = lambda p, t: hashlib.sha1(open(p, "rb").read() + b"\0" + open(t, "rb").read()).hexdigest()
    keys = [digest(p, t) for p, t in jobs]
    todo = {k: job for k, job in zip(keys, jobs) if k not in cache}
    with ThreadPoolExecutor(workers) as ex:
        for k, s in zip(todo, ex.map(lambda job: score_pair(usalign, *job)["tm_target"], todo.values())):
            cache[k] = s
    if cache_path and todo:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    return [cache[k] for k in keys]


def describe_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    cfg = ckpt["config"]
    with torch.device("meta"):
        n = sum(p.numel() for p in A2Holo(**cfg).parameters())
    plm = dict(ankh3="Ankh3", esm2="ESM-2", esmc="ESM-C")[cfg.get("plm", "ankh3")]
    stream = "none" if cfg["no_plm"] else "structure copy" if cfg.get("seq_from_structure") else plm
    attention = "—" if cfg["no_plm"] else cfg.get("attention", "joint")
    return dict(stream=stream, blocks=f"{cfg['depth_double']}+{cfg['depth_single']}", attention=attention, params=n,
                epoch=ckpt.get("epoch"))


def paired(a, b, lower_better=True):
    """Per-pair a − b over pairs where both are defined: median, better/tie/worse for a, Wilcoxon p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if not d.size:
        return dict(n=0)
    s = -1 if lower_better else 1
    p = float(wilcoxon(d).pvalue) if (d != 0).any() else NAN
    return dict(n=int(d.size), median=float(np.median(d)), mean=float(d.mean()), better=int((s * d > 0).sum()),
                tie=int((d == 0).sum()), worse=int((s * d < 0).sum()), p=p)


def fin(v):
    v = np.asarray(v, float)
    return v[np.isfinite(v)]


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


def parse_specs(specs, what):
    out = {}
    for s in specs:
        if "=" not in s:
            sys.exit(f"--{what} {s}: expected name=value")
        k, v = s.split("=", 1)
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=os.path.join(ROOT, "data/apollodiff_splitA.json"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--ref_dir", required=True, help="directory with target/ and apo/ <pair>.pdb")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PDB_DIR[,RESULTS_JSON]")
    ap.add_argument("--checkpoint", action="append", default=[], metavar="NAME=BEST_MODEL_PT")
    ap.add_argument("--baseline", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--size", action="append", default=[], metavar="SMALL=LARGE")
    ap.add_argument("--interaction", action="append", default=[], metavar="A_LARGE,A_SMALL,B_LARGE,B_SMALL")
    ap.add_argument("--displacement", action="append", default=[], metavar="A=B")
    ap.add_argument("--data_dir", help="pair tensors of the train split, for the (θ, τ) reference")
    ap.add_argument("--usalign", default=os.environ.get("USALIGN", "USalign"))
    ap.add_argument("--workers", type=int, default=4, help="parallel USalign processes")
    ap.add_argument("--tm_cache", help="JSON cache of TM-scores, keyed by file content")
    ap.add_argument("--out_json")
    args = ap.parse_args()

    with open(args.splits) as f:
        splits = json.load(f)
    ids = splits[args.split]
    models = {k: v.split(",") for k, v in parse_specs(args.model, "model").items()}
    main_model = next(iter(models))
    ckpts = {k: describe_checkpoint(v) for k, v in parse_specs(args.checkpoint, "checkpoint").items()}
    baselines = parse_specs(args.baseline, "baseline")
    samples = {b: sorted((os.path.basename(p) for p in glob.glob(os.path.join(d, "sample*")) if os.path.isdir(p)),
                         key=lambda s: int(re.sub(r"\D", "", s) or 0)) for b, d in baselines.items()}
    if not all(samples.values()):
        sys.exit(f"no sample* directories in {[baselines[b] for b, s in samples.items() if not s]}")
    outliers, tt_ref = outlier_cells(args.data_dir, splits["train"]) if args.data_dir else (None, None)

    # every structure as (name, pair) -> path; baseline samples are named <baseline>/<k>
    paths = {}
    for pid in ids:
        paths["apo", pid] = os.path.join(args.ref_dir, "apo", f"{pid}.pdb")
        paths["holo", pid] = os.path.join(args.ref_dir, "target", f"{pid}.pdb")
        for m, spec in models.items():
            paths[m, pid] = os.path.join(spec[0], f"{pid}.pdb")
        for b, d in baselines.items():
            for k, s in enumerate(samples[b]):
                paths[f"{b}/{k}", pid] = os.path.join(d, s, f"{pid}.pdb")
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        sys.exit(f"{len(missing)} PDB files missing, e.g. {missing[:3]}")
    scored = [key for key in paths if key[0] != "holo"]
    tm = dict(zip(scored, tm_scores([(paths[key], paths["holo", key[1]]) for key in scored],
                                    args.usalign, args.tm_cache, args.workers)))

    refs, S, X = {}, {}, {}
    for pid in ids:
        apo, apo_cb = read_pdb(paths["apo", pid])
        holo, holo_cb = read_pdb(paths["holo", pid])
        valid, windows = masks(apo, holo)
        refs[pid] = ref = dict(apo=apo, holo=holo, valid=valid, windows=windows, rg_apo=radius_of_gyration(apo),
                               apo_seg=bond_lengths(apo) < GAP, holo_keep=bond_lengths(holo) <= GAP)
        for name in sorted({key[0] for key in paths}):
            x, cb = (holo, holo_cb) if name == "holo" else (apo, apo_cb) if name == "apo" else read_pdb(paths[name, pid])
            if len(x) != len(holo):
                sys.exit(f"{paths[name, pid]}: {len(x)} residues, holo has {len(holo)}")
            X[name, pid] = x
            S[name, pid] = s = geometry(x, cb, ref, outliers)
            if name != "holo":
                s.update(ca=rmsd(x, holo), tm=tm[name, pid], lddt=ca_lddt(x, holo))
    for m, spec in models.items():
        if len(spec) > 1:
            with open(spec[1]) as f:
                r = json.load(f)
            cb = dict(zip(r["pair_ids"], r["cb_rmsd"]))
            for pid in ids:
                S[m, pid]["cb"] = NAN if cb[pid] is None else float(cb[pid])

    for (name, pid), g in S.items():
        g["new_breaks"] = g["breaks"] - S["apo", pid]["breaks"]
        g["dtm"] = (1.0 if name == "holo" else g["tm"]) - S["apo", pid]["tm"]

    pick = {}
    for b in baselines:
        for pid in ids:
            ss = [S[f"{b}/{k}", pid] for k in range(len(samples[b]))]
            pick[b, "sample0", pid] = 0
            pick[b, "geometry", pid] = min(range(len(ss)), key=lambda k: (ss[k]["n05"], ss[k]["clash"]))
            pick[b, "oracle", pid] = min(range(len(ss)), key=lambda k: ss[k]["ca"])
            pick[b, "tm_best", pid] = max(range(len(ss)), key=lambda k: ss[k]["tm"])

    def row(name, rule=None):
        if rule is None:
            return [S[name, pid] for pid in ids]
        return [S[f"{name}/{pick[name, rule, pid]}", pid] for pid in ids]

    col = lambda r, key: np.array([s.get(key, NAN) for s in r], float)
    med = lambda r, key: float(np.median(fin(col(r, key)))) if fin(col(r, key)).size else NAN

    def acc(r):
        ca, tmv, ld = col(r, "ca"), col(r, "tm"), col(r, "lddt")
        return [f"{np.median(ca):.3f}", f"{ca.mean():.3f}", f"{100 * np.mean(ca < 2):.1f}%",
                f"{100 * np.mean(ca < 5):.1f}%", f"{np.median(tmv):.3f}", f"{ld.mean():.3f}"]

    geo = lambda r: [f"{med(r, 'disp'):.2f}", f"{100 * med(r, 'v05'):.1f}", f"{med(r, 'clash'):.1f}"]
    acc_h = ["Cα med", "Cα mean", "<2 Å", "<5 Å", "TM med", "lDDT mean"]
    geo_h = ["displacement from apo Å", "Cα–Cα violation %", "Cα clash"]
    n_s = {b: len(samples[b]) for b in baselines}
    pose = lambda b, rule: "1" if rule == "sample0" else f"1 of {n_s[b]}"

    print(f"# {os.path.basename(args.splits)}, {args.split} ({len(ids)} pairs)\n")
    print("## Accuracy and geometry\n")
    rows = [["apo input", "1"] + acc(row("apo")) + geo(row("apo")),
            ["holo (geometry only)", "—"] + ["—"] * 6 + geo(row("holo"))]
    rows += [[f"{b} ({RULE_LABEL['geometry']})", pose(b, "geometry")] + acc(row(b, "geometry")) + geo(row(b, "geometry"))
             for b in baselines]
    rows += [[main_model, "1"] + acc(row(main_model)) + geo(row(main_model))]
    table(["Method", "poses"] + acc_h + geo_h, rows)

    print("## Accuracy by selection rule\n")
    rows = [["apo input", "1"] + acc(row("apo"))]
    rows += [[f"{b} ({RULE_LABEL[rule]})", pose(b, rule)] + acc(row(b, rule)) for b in baselines for rule in RULES]
    rows += [[main_model, "1"] + acc(row(main_model))]
    table(["Method", "poses"] + acc_h, rows)

    print("## Geometry (median over pairs)\n")
    f = lambda v, fmt: "—" if not np.isfinite(v) else format(v, fmt)
    hdr = ["Method", "displacement from apo Å", "Cα–Cα violation %", "|d−3.8| > 0.2 %", "Cα clash"]
    hdr += ["(θ,τ) outlier %"] if outliers is not None else []
    hdr += ["ΔRg Å", "Cα–Cβ violation %", "Cβ–Cβ clash"]
    rows = []
    for lab, r in ([("apo input", row("apo")), ("holo", row("holo"))]
                   + [(f"{b} ({RULE_LABEL['geometry']})", row(b, "geometry")) for b in baselines]
                   + [(main_model, row(main_model))]):
        cells_ = [lab, f(med(r, "disp"), ".2f"), f(100 * med(r, "v05"), ".1f"), f(100 * med(r, "v02"), ".1f"),
                  f(med(r, "clash"), ".1f")]
        cells_ += [f(100 * med(r, "theta_tau_out"), ".1f")] if outliers is not None else []
        cells_ += [f(med(r, "drg"), "+.2f"), f(100 * med(r, "cacb"), ".1f"), f(med(r, "cbclash"), ".1f")]
        rows.append(cells_)
    table(hdr, rows)
    if tt_ref:
        print(f"(θ,τ) reference: {tt_ref['windows']} windows of {tt_ref['train_pairs']} train holo structures; "
              f"{tt_ref['outlier_cells']} of {tt_ref['cells']} cells (count <= {tt_ref['count_threshold']}) "
              f"hold {100 * tt_ref['outlier_mass']:.3f} % of them.\n")

    print("## Backbone against the apo input (baselines: TM-best sample; medians over pairs)\n")
    hdr = ["Method", "Cα–Cα violation % (bonds holo keeps)", "largest gap Å", "new chain breaks",
           "pairs with no new break", "TM gain over apo"]
    backbone = {}
    rows = []
    for lab, r in ([("apo input", row("apo")), ("holo", row("holo"))]
                   + [(f"{b} (TM-best of {n_s[b]})", row(b, "tm_best")) for b in baselines]
                   + [(m, row(m)) for m in models]):
        nb_ = col(r, "new_breaks")
        backbone[lab] = dict(v05_holo=med(r, "v05_holo"), max_gap=float(np.nanmax(col(r, "max_gap"))),
                             new_breaks=float(np.median(nb_)), no_new_break=int((nb_ <= 0).sum()), dtm=med(r, "dtm"))
        c = backbone[lab]
        rows.append([lab, f(100 * c["v05_holo"], ".1f"), f(c["max_gap"], ".1f"), f"{c['new_breaks']:.1f}",
                     f"{c['no_new_break']}/{len(ids)}", f"{c['dtm']:+.3f}"])
    table(hdr, rows)

    def sign_test(a_, b_):
        d_ = col(a_, "new_breaks") - col(b_, "new_breaks")
        w, t, l_ = int((d_ < 0).sum()), int((d_ == 0).sum()), int((d_ > 0).sum())
        return dict(fewer=w, tie=t, more=l_, p=float(binomtest(w, w + l_).pvalue) if w + l_ else NAN)

    rows = []
    for lab, other in ([(f"{b} (TM-best)", row(b, "tm_best")) for b in baselines] + [("holo", row("holo"))]):
        t = backbone[f"{main_model} vs {lab}"] = sign_test(row(main_model), other)
        rows.append([lab, str(t["fewer"]), str(t["tie"]), str(t["more"]), f"{t['p']:.1e}"])
    print(f"New chain breaks, {main_model} against each, per pair (exact two-sided sign test, ties left out):\n")
    table(["Against", f"{main_model} fewer", "tie", f"{main_model} more", "p"], rows)

    print("## Models\n")
    hdr = ["Model"] + (["second stream", "blocks", "attention", "parameters", "epoch"] if ckpts else [])
    hdr += ["Cα med", "Cα mean", "<5 Å", "Cβ med", "TM med", "lDDT mean", "displacement from apo Å"]
    rows = []
    for m in models:
        r = row(m)
        ca, cb = col(r, "ca"), col(r, "cb")
        c = ckpts.get(m)
        about = [c["stream"], c["blocks"], c["attention"], f"{c['params']:,}", str(c["epoch"])] if c else ["—"] * 5
        rows.append([m] + (about if ckpts else [])
                    + [f"{np.median(ca):.3f}", f"{ca.mean():.3f}", f"{100 * np.mean(ca < 5):.1f}%",
                       f(np.nanmedian(cb) if fin(cb).size else NAN, ".3f"),
                       f"{np.median(col(r, 'tm')):.3f}", f"{col(r, 'lddt').mean():.3f}", f"{med(r, 'disp'):.2f}"])
    table(hdr, rows)

    stat = lambda t, fmt=".3f": f"{t['median']:+{fmt}} ({t['better']}/{t['worse']}, p {t['p']:.1e})"
    pairwise = {}
    if baselines:
        print(f"## {main_model} − baseline, per pair: median Δ (better/worse, Wilcoxon p)\n")
        rows = []
        a = row(main_model)
        for b in baselines:
            pairwise[b] = {}
            for rule in RULES:
                o = row(b, rule)
                t = pairwise[b][rule] = {k: paired(col(a, k), col(o, k), lower_better=(k == "ca"))
                                         for k in ("ca", "tm", "lddt")}
                rows.append([b, RULE_LABEL[rule]] + [stat(t[k]) for k in ("ca", "tm", "lddt")])
            best = [max(S[f"{b}/{k}", pid]["tm"] for k in range(n_s[b])) for pid in ids]
            t = pairwise[b]["tm_best"] = paired(col(a, "tm"), best, lower_better=False)
            rows.append([b, f"TM-best of {n_s[b]}", "—", stat(t), "—"])
        table(["Baseline", "sample kept", "Cα", "TM", "lDDT"], rows)
        for k, lab in (("ca", "Cα RMSD"), ("tm", "TM-score"), ("lddt", "lDDT")):
            ts = [pairwise[b][rule][k] for b in baselines for rule in RULES]
            print(f"{lab}: {main_model} better in {min(t['better'] for t in ts)}–{max(t['better'] for t in ts)} of "
                  f"{len(ids)} pairs over the {len(ts)} baseline × rule comparisons, p ≤ {max(t['p'] for t in ts):.1e}")
        ts = [pairwise[b]["tm_best"] for b in baselines]
        print(f"TM-score against each baseline's TM-best: {main_model} better in {min(t['better'] for t in ts)}–"
              f"{max(t['better'] for t in ts)} of {len(ids)} pairs, p ≤ {max(t['p'] for t in ts):.1e}\n")

    sizes = {}
    if args.size:
        print("## Size within a row, larger − smaller, per pair: median Δ (larger better/worse, Wilcoxon p)\n")
        rows = []
        for small, large in parse_specs(args.size, "size").items():
            t = {k: paired(col(row(large), k), col(row(small), k)) for k in ("ca", "cb")}
            sizes[f"{small}->{large}"] = t
            rows.append([f"{small} → {large}"] + [stat(t[k]) if t[k]["n"] else "—" for k in ("ca", "cb")])
        table(["Comparison", "Cα", "Cβ"], rows)

    if args.interaction:
        print("## Size steps compared, (a_large − a_small) − (b_large − b_small), per pair: median Δ "
              "(a's step better/worse, Wilcoxon p)\n")
        rows = []
        for spec in args.interaction:
            al, as_, bl, bs = spec.split(",")
            t = {k: paired(col(row(al), k) - col(row(as_), k), col(row(bl), k) - col(row(bs), k)) for k in ("ca", "cb")}
            sizes[f"({al}-{as_})-({bl}-{bs})"] = t
            rows.append([f"({al} − {as_}) − ({bl} − {bs})"] + [stat(t[k]) if t[k]["n"] else "—" for k in ("ca", "cb")])
        table(["Comparison", "Cα", "Cβ"], rows)

    if args.displacement:
        print("## Displacement from apo, a − b, per pair: median Δ Å (a moves further/less, Wilcoxon p)\n")
        rows = []
        for a_, b_ in parse_specs(args.displacement, "displacement").items():
            t = sizes[f"displacement {a_}-{b_}"] = paired(col(row(a_), "disp"), col(row(b_), "disp"), lower_better=False)
            rows.append([f"{a_} − {b_}", stat(t)])
        table(["Comparison", "displacement"], rows)

    matched = {}
    if baselines:
        print("## Baselines by selection rule: Cα median / Cα–Cα violation %\n")
        rows = [[b] + [f"{med(row(b, rule), 'ca'):.3f} / {100 * med(row(b, rule), 'v05'):.1f}" for rule in RULES]
                for b in baselines]
        table(["Baseline"] + [RULE_LABEL[r] for r in RULES], rows)

        print(f"## Displacement matched: {main_model} − baseline sample 0, Cα–Cα violation rate per pair\n")
        rows = []
        for b in baselines:
            va, vb = [], []
            for pid in ids:
                sa, sb = S[main_model, pid], S[f"{b}/0", pid]
                a, o = sa["v05"], sb["v05"]
                if sa["disp"] > sb["disp"]:
                    a = v05_after_shrink(X[main_model, pid], refs[pid], sb["disp"])
                elif sb["disp"] > sa["disp"]:
                    o = v05_after_shrink(X[f"{b}/0", pid], refs[pid], sa["disp"])
                va.append(a)
                vb.append(o)
            t = matched[b] = paired(va, vb)
            rows.append([b, str(t["n"]), f"{100 * t['median']:+.1f}", f"{100 * t['mean']:+.1f}",
                         f"{t['better']}/{t['tie']}/{t['worse']}", f"{t['p']:.1e}"])
        table(["Baseline", "n", "Δ median pp", "Δ mean pp", "better/tie/worse", "p"], rows)

    if args.out_json:
        clean = lambda v: None if isinstance(v, float) and not np.isfinite(v) else v
        out = dict(splits=args.splits, split=args.split, pair_ids=ids, models={m: s for m, s in models.items()},
                   checkpoints=ckpts, baselines=baselines, samples=samples, theta_tau_reference=tt_ref,
                   per_pair={pid: {name: {k: clean(v) for k, v in S[name, pid].items()}
                                   for (name, p) in S if p == pid} for pid in ids},
                   picks={b: {rule: [pick[b, rule, pid] for pid in ids] for rule in RULES + ("tm_best",)}
                          for b in baselines},
                   pairwise=pairwise, sizes=sizes, displacement_matched=matched, backbone=backbone)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=1, default=lambda v: clean(float(v)))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
