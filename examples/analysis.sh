#!/bin/bash
# Mechanism analyses on one trained checkpoint, as run for the paper: a gate, then the attention,
# representation and intervention measurements at t = 0.5 and t = 0.
#
#   examples/analysis.sh <checkpoint_dir> [label]
#
# <checkpoint_dir> holds best_model.pt. Needs a GPU. Writes under $ANALYSIS_OUT/<label>
# (default outputs/analysis/<label>).
#   TLIST="0.5 0"   flow times to analyse (the analysed input is (1 - t) * apo + t * holo)
#   CONTROLS=1      also run the random-weights control at each time
set -euo pipefail
cd "$(dirname "$0")/.."

CKDIR="${1:?usage: examples/analysis.sh <checkpoint_dir> [label]}"
LABEL="${2:-$(basename "$CKDIR")}"
export ANALYSIS_OUT="${ANALYSIS_OUT:-outputs/analysis}/$LABEL"
TLIST="${TLIST:-0.5 0}"
mkdir -p "$ANALYSIS_OUT"

# 1. Gate: at t = 0 the analysis forward pass must reproduce, pair by pair, the one-step prediction
#    of the evaluation script. Nothing else runs if it does not.
python scripts/evaluate.py --checkpoint "$CKDIR/best_model.pt" --num_steps 1 --output "$ANALYSIS_OUT/nfe1_results.json"
python analysis/validate_oneshot_readout.py --ckdir "$CKDIR" \
    --eval_json "$ANALYSIS_OUT/nfe1_results.json" --out_dir "$ANALYSIS_OUT/gate"

# 2. Apo-geometry baselines for the moving-residue overlap; no model involved.
python analysis/geometric_baselines.py --max_pairs 150 --tag geometric_baselines

# 3. The measurements at each flow time, with the paper's pair counts.
run() { python analysis/run.py --script "analysis/$1" --t "$T" --checkpoint "$LABEL=$CKDIR" "${@:2}"; }
for T in $TLIST; do
    run emit_recv_reference_set.py             --max_pairs 150 --tag "t$T/emit_recv_reference_set"
    run attention_conditional.py               --max_pairs 200 --tag "t$T/attention_conditional"
    run update_alignment.py                    --max_pairs 150 --tag "t$T/update_alignment_reference_set"
    run update_alignment.py --blocks all       --max_pairs 150 --tag "t$T/update_alignment_allblocks"
    run decodability_relative_reference_set.py --max_pairs 150 --tag "t$T/decodability_relative_reference_set"
    run quadrant_intervention.py               --max_pairs 40  --tag "t$T/quadrant_intervention"
    run alpha_scaling_reference_set.py         --max_pairs 200 --tag "t$T/alpha_scaling"
    if [ "${CONTROLS:-0}" = 1 ]; then
        ANALYSIS_OUT="$ANALYSIS_OUT/t$T" python analysis/run.py --script analysis/emit_recv_controls.py \
            --t "$T" --mode random --max_pairs 150 --random_arch "$CKDIR"
    fi
done
echo "done: $ANALYSIS_OUT"
