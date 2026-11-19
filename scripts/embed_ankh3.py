"""Ankh3-large embeddings of each pair's sequence, read from the node features so lengths always match.

    python scripts/embed_ankh3.py --data_dir data/d3pm/pairs --output_dir data/d3pm/ankh3

Writes <pair>.pt = {embeddings [N, 1536], global_token [1536]}: the T5 encoder's last hidden state
for "[NLU]" + sequence, per residue and at the [NLU] prefix. fp32; the released embeddings were
computed on the CPU.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2holo.data import DATA_DIR, EMB_DIR  # noqa: E402
from a2holo.pdb import sequence  # noqa: E402


@torch.no_grad()
def embed(tokenizer, model, seq, device):
    """Per-residue embeddings [N, d] and the [NLU] token [d]; the closing </s> is dropped."""
    tokens = tokenizer("[NLU]" + seq, return_tensors="pt", add_special_tokens=True)
    hidden = model(input_ids=tokens["input_ids"].to(device),
                   attention_mask=tokens["attention_mask"].to(device)).last_hidden_state[0].cpu()
    return hidden[1:-1], hidden[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--output_dir", default=EMB_DIR)
    ap.add_argument("--model_name", default="ElnaggarLab/ankh3-large")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    n_tokens = len(tokenizer("[NLU]ACDEF", return_tensors="pt", add_special_tokens=True)["input_ids"][0])
    if n_tokens != 7:
        raise SystemExit(f"{args.model_name}: '[NLU]ACDEF' gives {n_tokens} tokens, expected one per residue + 2")
    device = torch.device(args.device)
    model = T5EncoderModel.from_pretrained(args.model_name).to(device).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pt_file in tqdm(sorted(Path(args.data_dir).glob("*.pt")), desc="Ankh3"):
        out = output_dir / f"{pt_file.stem}.pt"
        if out.exists() and not args.overwrite:
            continue
        seq = sequence(torch.load(pt_file, weights_only=False).x)
        embeddings, global_token = embed(tokenizer, model, seq, device)
        if embeddings.shape != (len(seq), model.config.d_model):
            raise ValueError(f"{pt_file.stem}: embeddings {tuple(embeddings.shape)} for {len(seq)} residues")
        torch.save({"embeddings": embeddings, "global_token": global_token}, out)


if __name__ == "__main__":
    main()
