"""ESM-2 650M or ESM-C 300M embeddings of each pair's sequence, read from the node features.

    python scripts/embed_esm.py --model esm2 --data_dir data/d3pm/pairs    # -> data/d3pm/esm2
    python scripts/embed_esm.py --model esmc --data_dir data/d3pm/pairs    # -> data/d3pm/esmc

Same format as embed_ankh3.py: last hidden state per residue, and at position 0 (<cls> for ESM-2, BOS
for ESM-C) as the global token; closing token dropped. Models: facebook/esm2_t33_650M_UR50D
(transformers), esmc_300m (esm).
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2holo.data import DATA_DIR, emb_dir_for  # noqa: E402
from a2holo.model import PLM_DIMS  # noqa: E402
from a2holo.pdb import sequence  # noqa: E402


class ESM2:
    def __init__(self, name, device, dtype):
        from transformers import EsmModel, EsmTokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(name)
        self.model = EsmModel.from_pretrained(name, add_pooling_layer=False, torch_dtype=dtype).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, seq):
        tokens = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        if tokens["input_ids"].shape[1] != len(seq) + 2:
            raise ValueError(f"{tokens['input_ids'].shape[1]} tokens for {len(seq)} residues")
        return self.model(input_ids=tokens["input_ids"].to(self.device),
                          attention_mask=tokens["attention_mask"].to(self.device)).last_hidden_state[0].float().cpu()


class ESMC:
    def __init__(self, name, device, dtype):
        from esm.models.esmc import ESMC as Model
        # Loaded on the CPU and then moved, since from_pretrained casts to bfloat16 on a GPU.
        self.model = Model.from_pretrained(name, device=torch.device("cpu")).to(device=device, dtype=dtype).eval()

    @torch.no_grad()
    def __call__(self, seq):
        from esm.sdk.api import ESMProtein, LogitsConfig
        out = self.model.logits(self.model.encode(ESMProtein(sequence=seq)),
                                LogitsConfig(sequence=True, return_embeddings=True))
        return out.embeddings[0].float().cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("esm2", "esmc"), required=True)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--output_dir", help="default: $EMB_DIR, else data/d3pm/<model>")
    ap.add_argument("--model_name", help="default: facebook/esm2_t33_650M_UR50D for esm2, esmc_300m for esmc; "
                                         "a local copy of the weights works too")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    if args.model == "esm2":
        embedder = ESM2(args.model_name or "facebook/esm2_t33_650M_UR50D", device, dtype)
    else:
        embedder = ESMC(args.model_name or "esmc_300m", device, dtype)
    dim = PLM_DIMS[args.model]

    output_dir = Path(args.output_dir or emb_dir_for(args.model))
    output_dir.mkdir(parents=True, exist_ok=True)
    for pt_file in tqdm(sorted(Path(args.data_dir).glob("*.pt")), desc=args.model):
        out = output_dir / f"{pt_file.stem}.pt"
        if out.exists() and not args.overwrite:
            continue
        seq = sequence(torch.load(pt_file, weights_only=False).x)
        hidden = embedder(seq)
        if hidden.shape != (len(seq) + 2, dim):
            raise ValueError(f"{pt_file.stem}: hidden states {tuple(hidden.shape)} for {len(seq)} residues")
        torch.save({"embeddings": hidden[1:-1].clone(), "global_token": hidden[0].clone()}, out)


if __name__ == "__main__":
    main()
