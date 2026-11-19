#!/bin/bash
# Ablations of the reported recipe: scripts/train.py (seed 1) with one component changed.
#
#   examples/ablations.sh no-plm            # no sequence stream (and so no double-stream blocks)
#   examples/ablations.sh no-node           # node features zeroed, capacity unchanged
#   examples/ablations.sh no-global-token   # global token zeroed, per-residue embeddings kept
#   examples/ablations.sh depth 4 8 16      # equal double- and single-stream depth, one run per value
set -euo pipefail
cd "$(dirname "$0")/.."
train() { python scripts/train.py --seed 1 --out_dir "checkpoints/ablation_$1" "${@:2}"; }
case "${1:-}" in
  no-plm)          train no_plm --no_plm --depth_double 0 ;;
  no-node)         train no_node --no_node_features ;;
  no-global-token) train no_global_token --no_global_token ;;
  depth)
    shift
    for d in ${@:-2 4 6 8 10 14 16}; do train "depth$d" --depth_double "$d" --depth_single "$d"; done ;;
  *)
    echo "usage: examples/ablations.sh {no-plm|no-node|no-global-token|depth [N ...]}" >&2
    exit 1 ;;
esac
