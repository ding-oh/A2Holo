#!/bin/bash
# From SBAlign's preprocessed Cα-only D3PM pairs (see README.md, Data) to the model inputs:
# Cα+Cβ tensors in data/d3pm/pairs and Ankh3-large embeddings in data/d3pm/ankh3 ($DATA_DIR, $EMB_DIR).
# $SBALIGN is the SBAlign checkout (default data/sbalign). ESM=1 also writes ESM-2 and ESM-C embeddings
# (data/d3pm/esm2, data/d3pm/esmc).
set -euo pipefail
cd "$(dirname "$0")/.."
SBALIGN="${SBALIGN:-data/sbalign}"

python scripts/add_cb.py --sbalign_dir "$SBALIGN" \
    --pt_dir "$SBALIGN/data/processed/d3pm/resolution=c_alpha_centered_conf" \
    --cif_dir "$SBALIGN/data/raw/d3pm/conformations" \
    --output_dir "${DATA_DIR:-data/d3pm/pairs}"
python scripts/embed_ankh3.py --data_dir "${DATA_DIR:-data/d3pm/pairs}" --output_dir "${EMB_DIR:-data/d3pm/ankh3}"
if [ "${ESM:-0}" = 1 ]; then
    for m in esm2 esmc; do
        python scripts/embed_esm.py --model $m --data_dir "${DATA_DIR:-data/d3pm/pairs}" --output_dir data/d3pm/$m
    done
fi
