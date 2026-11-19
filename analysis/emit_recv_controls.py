#!/usr/bin/env python3
r"""Two controls for the emitted/received attention measurement.

  --mode random   The same measurement on randomly initialised weights: same
                  architecture, same inputs, one model per torch seed.  Tests whether
                  the overlap with the moving residues is learned or a property of the
                  token layout.
  --mode seeds    The same measurement on independent trainings of one configuration
                  that differ only in seed.

The measurement is not copied: this imports emit_recv_reference_set and replaces its
model list (and, for --mode random, its loader), so the control and the result it
controls for come from the same code path.

The models come from --seed_models / --random_arch, not from run.py's --checkpoint.
Defaults: the five seed trainings under checkpoints/ for --mode seeds, and the seed-1
checkpoint's config for --mode random.  Writes emit_recv_seeds*.tsv or
emit_recv_random*.tsv under common.OUT_DIR (set ANALYSIS_OUT to separate runs).

    python analysis/run.py --script analysis/emit_recv_controls.py --t 0.5 \
        --mode random --max_pairs 150
    python analysis/run.py --script analysis/emit_recv_controls.py --t 0.5 \
        --mode seeds --max_pairs 150 \
        --seed_models seed1=checkpoints/a2holo_seed1 \
        --seed_models seed2=checkpoints/a2holo_seed2 \
        --seed_models seed3=checkpoints/a2holo_seed3 \
        --seed_models seed4=checkpoints/a2holo_seed4 \
        --seed_models seed5=checkpoints/a2holo_seed5
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch                                          # noqa: E402
from common import ROOT                               # noqa: E402
from a2holo.model import A2Holo                       # noqa: E402
import emit_recv_reference_set as ER                  # noqa: E402

P = os.path.join(ROOT, "checkpoints", "a2holo_")
SEED_MODELS = [
    ("seed1", f"{P}seed1"),
    ("seed2", f"{P}seed2"),
    ("seed3", f"{P}seed3"),
    ("seed4", f"{P}seed4"),
    ("seed5", f"{P}seed5"),
]
# The architecture the random control instantiates.  Its checkpoint is read ONLY for its
# config; none of its weights are used.
RANDOM_ARCH = f"{P}seed1"
RANDOM_SEEDS = [0, 1, 2]


def make_random_loader(seed):
    """Build the checkpoint's architecture as common.load_model does, but never load its weights."""
    def load(ckpt_path, device):
        config = torch.load(ckpt_path, map_location="cpu", weights_only=True)["config"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = A2Holo(**config).to(device)
        model.eval()
        n_par = sum(p.numel() for p in model.parameters())
        print(f"[random init] seed={seed} from {os.path.basename(os.path.dirname(ckpt_path))} "
              f"config={config} params={n_par}", flush=True)
        return model, config["depth_double"]
    return load


def parse_models(specs):
    out = []
    for s in specs:
        if "=" not in s:
            sys.exit(f"--seed_models wants LABEL=CHECKPOINT_DIR, got {s!r}")
        label, ckdir = s.split("=", 1)
        out.append((label, ckdir))
    return out


ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--mode", choices=("random", "seeds"), required=True)
ap.add_argument("--max_pairs", type=int, default=40)
ap.add_argument("--seed_models", action="append", default=None, metavar="LABEL=CKDIR",
                help="repeat once per checkpoint; default = the five seed checkpoints "
                     "under checkpoints/")
ap.add_argument("--random_arch", default=None, metavar="CKDIR",
                help="checkpoint whose config the random control uses; "
                     "no weight of it is read.  Default = the seed-1 checkpoint")
ap.add_argument("--random_seeds", type=int, nargs="+", default=None,
                help="torch seeds for the random initialisations (default 0 1 2)")
args = ap.parse_args()

seed_models = parse_models(args.seed_models) if args.seed_models else SEED_MODELS
random_arch = args.random_arch or RANDOM_ARCH
random_seeds = args.random_seeds if args.random_seeds is not None else RANDOM_SEEDS

if args.mode == "seeds":
    ER.MODELS = seed_models
    tag = "emit_recv_seeds"
else:
    # One entry per random seed, all pointing at the same architecture.
    ER.MODELS = [(f"random init s{s}", random_arch) for s in random_seeds]
    tag = "emit_recv_random"

    _loaders = iter([make_random_loader(s) for s in random_seeds])

    def _dispatch(path, device):
        return next(_loaders)(path, device)
    ER.load_model = _dispatch

missing = [c for _, c in ER.MODELS
           if not os.path.exists(os.path.join(c, "best_model.pt"))]
if missing:
    sys.exit("missing checkpoints, refusing to run a partial control: %s" % missing)

print(f"[control] mode={args.mode}  t={ER.T_VAL}  max_pairs={args.max_pairs}  "
      f"out={ER.OUT}", flush=True)
print(f"[control] models={ER.MODELS}", flush=True)
sys.argv = ["emit_recv_reference_set.py", "--max_pairs", str(args.max_pairs), "--tag", tag]
ER.main()
