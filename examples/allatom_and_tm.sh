#!/bin/bash
# TM-score of the predicted backbones and all-atom reconstruction with cg2all.
#
#   examples/allatom_and_tm.sh <checkpoint.pt> [out_dir]
#
# USalign ($USALIGN, or on PATH) and cg2all's convert_cg2all ($CONVERT_CG2ALL, or on PATH) are
# installed separately.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:?usage: examples/allatom_and_tm.sh <checkpoint.pt> [out_dir]}"
OUT="${2:-outputs/structures}"

# Predicted, holo and apo structures of every test pair (10 Euler steps, EMA weights).
python scripts/evaluate.py --checkpoint "$CKPT" --pdb_dir "$OUT/pdb" --output "$OUT/eval.json"
# TM-score against holo, for the prediction and for the unchanged apo input.
python scripts/tm_score.py --pred_dir "$OUT/pdb/pred" --target_dir "$OUT/pdb/target" --out_json "$OUT/tm_a2holo.json" --name a2holo
python scripts/tm_score.py --pred_dir "$OUT/pdb/apo" --target_dir "$OUT/pdb/target" --out_json "$OUT/tm_apo.json"
# All-atom models from the predicted, apo and holo Cα traces.
python scripts/allatom_cg2all.py --checkpoint "$CKPT" --output_dir "$OUT/allatom"
