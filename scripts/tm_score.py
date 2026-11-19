"""TM-score of predicted against target structures with USalign.

    python scripts/tm_score.py --pred_dir D/pred --target_dir D/target --out_json D/tm_a2holo.json

Scores every <pair>.pdb present in both directories (evaluate.py --pdb_dir writes pred/, target/ and
apo/; USalign reads their Cα atoms). tm_target is normalised by the target length, the reported
TM-score; tm_pred by the prediction length. Writes per-pair scores and a summary.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np


# "TM-score= 0.76543 (normalized by length of Structure_1: L=150, d0=4.47)"; Structure_1 is the prediction.
TM_LINE_RE = re.compile(r"TM-score=\s*([0-9.]+)\s*\(normalized by length of Structure_?([12])", re.IGNORECASE)


def score_pair(usalign, pred_pdb, target_pdb):
    result = subprocess.run([usalign, pred_pdb, target_pdb, "-mol", "prot", "-outfmt", "-1"],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"USalign failed for {pred_pdb} vs {target_pdb}:\n"
                           f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    scores = {f"tm_struct{m.group(2)}": float(m.group(1)) for m in TM_LINE_RE.finditer(result.stdout)}
    if "tm_struct1" not in scores or "tm_struct2" not in scores:
        raise RuntimeError(f"Could not parse TM-scores from USalign output for {pred_pdb} vs {target_pdb}. "
                           f"Output:\n{result.stdout}")
    scores["tm_pred"] = scores["tm_struct1"]
    scores["tm_target"] = scores["tm_struct2"]
    return scores


def summarize(per_pair):
    tm_target = np.array([s["tm_target"] for s in per_pair.values()])
    tm_pred = np.array([s["tm_pred"] for s in per_pair.values()])
    return {
        "n_pairs": int(tm_target.size),
        "tm_target_median": float(np.median(tm_target)),
        "tm_target_mean": float(np.mean(tm_target)),
        "tm_target_std": float(np.std(tm_target)),
        "tm_target_min": float(np.min(tm_target)),
        "tm_target_max": float(np.max(tm_target)),
        "tm_target_pct_ge_0.5": float(np.mean(tm_target >= 0.5) * 100),
        "tm_target_pct_ge_0.7": float(np.mean(tm_target >= 0.7) * 100),
        "tm_target_pct_ge_0.9": float(np.mean(tm_target >= 0.9) * 100),
        "tm_pred_median": float(np.median(tm_pred)),
        "tm_pred_mean": float(np.mean(tm_pred)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--name", help="label recorded in the JSON (default: the name of --pred_dir)")
    ap.add_argument("--usalign", default=os.environ.get("USALIGN", "USalign"))
    args = ap.parse_args()
    name = args.name or os.path.basename(os.path.normpath(args.pred_dir))
    if not shutil.which(args.usalign):
        sys.exit(f"USalign binary not found at {args.usalign}")

    pdbs = lambda d: {f.removesuffix(".pdb") for f in os.listdir(d) if f.endswith(".pdb")}
    preds, targets = pdbs(args.pred_dir), pdbs(args.target_dir)
    for what, missing in (("targets without a prediction", targets - preds),
                          ("predictions without a target", preds - targets)):
        if missing:
            print(f"[warn] {len(missing)} {what} (first 5): {sorted(missing)[:5]}")

    per_pair, errors = {}, []
    for pid in sorted(preds & targets):
        try:
            per_pair[pid] = score_pair(args.usalign, os.path.join(args.pred_dir, f"{pid}.pdb"),
                                       os.path.join(args.target_dir, f"{pid}.pdb"))
        except Exception as e:
            errors.append(f"{pid}: {e}")

    summary = summarize(per_pair) if per_pair else {}
    summary["model"] = name
    summary["n_errors"] = len(errors)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"model": name, "pred_dir": args.pred_dir, "target_dir": args.target_dir,
                   "per_pair": per_pair, "summary": summary, "errors": errors}, f, indent=2)

    print(f"{name}: {summary.get('n_pairs', 0)} pairs, {len(errors)} errors")
    for k in ("tm_target_median", "tm_target_mean", "tm_target_std",
              "tm_target_pct_ge_0.5", "tm_target_pct_ge_0.7", "tm_target_pct_ge_0.9"):
        if k in summary:
            print(f"  {k}: {summary[k]:.4f}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
