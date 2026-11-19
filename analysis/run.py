"""Run one analysis script on chosen checkpoints at a chosen flow time.

    python analysis/run.py --script analysis/emit_recv_reference_set.py --t 0 \\
        --checkpoint seed1=checkpoints/a2holo_seed1 \\
        --max_pairs 150 --tag t0/emit_recv_reference_set

--checkpoint is LABEL=DIR, where DIR holds best_model.pt; repeat it to analyse several
models in one run. Every other argument is passed to the script. The script reads the
models and the time from common.MODELS and common.T_VAL, which are set here before it
starts.
"""
import argparse
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# allow_abbrev=False: a script's own options must not be taken as prefixes of these.
ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--script", required=True)
ap.add_argument("--t", type=float, required=True, help="flow time of the analysed input")
ap.add_argument("--checkpoint", action="append", default=[], help="LABEL=DIR, DIR holding best_model.pt")
args, rest = ap.parse_known_args()

import common  # noqa: E402

models = []
for spec in args.checkpoint:
    if "=" not in spec:
        sys.exit(f"--checkpoint needs LABEL=DIR, got {spec!r}")
    label, ckdir = spec.split("=", 1)
    if not os.path.exists(os.path.join(ckdir, "best_model.pt")):
        sys.exit(f"no best_model.pt in {ckdir}")
    models.append((label, ckdir))
common.MODELS = models
common.T_VAL = args.t

# Scripts write OUT_DIR/<tag>.tsv; a tag may name a subdirectory, which is created here.
for i, a in enumerate(rest):
    tag = rest[i + 1] if a == "--tag" and i + 1 < len(rest) else (a[len("--tag="):] if a.startswith("--tag=") else None)
    if tag:
        os.makedirs(os.path.dirname(os.path.join(common.OUT_DIR, tag)) or common.OUT_DIR, exist_ok=True)

sys.argv = [args.script] + rest
runpy.run_path(args.script, run_name="__main__")
