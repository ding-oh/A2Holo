"""D3PM apo-holo pairs with their protein language model embeddings.

Each pair is a PyG Data object from data_dir/<pair>.pt holding x [N, 145] node features, pos_0 /
pos_T [N, 3] apo / holo Cα, pos_0_cb / pos_T_cb [N, 3] apo / holo Cβ and cb_mask [N] (0 for
glycine's virtual Cβ). emb_dir/<pair>.pt holds per-residue embeddings [N, D] and the global token [D],
the first token ([NLU] for Ankh3, <cls>/BOS for ESM-2/ESM-C); D = PLM_DIMS[plm].
"""

import json
import os

import torch
from torch_geometric.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data/d3pm/pairs"))
def emb_dir_for(plm="ankh3"):
    return os.environ.get("EMB_DIR", os.path.join(ROOT, "data/d3pm", plm))


EMB_DIR = emb_dir_for("ankh3")
SPLITS = os.environ.get("SPLITS", os.path.join(ROOT, "data/d3pm_splits.json"))


def split_ids(splits_file, split):
    with open(splits_file) as f:
        return json.load(f)[split]


class PairDataset(Dataset):
    """emb_dim, when given, is the embedding width the model takes; files of another width fail."""

    def __init__(self, data_dir, emb_dir, pair_ids, emb_dim=None):
        super().__init__()
        self.data_dir, self.emb_dir, self.pair_ids, self.emb_dim = data_dir, emb_dir, list(pair_ids), emb_dim

    def len(self):
        return len(self.pair_ids)

    def get(self, idx):
        pid = self.pair_ids[idx]
        data = torch.load(os.path.join(self.data_dir, f"{pid}.pt"), weights_only=False)
        emb = torch.load(os.path.join(self.emb_dir, f"{pid}.pt"), weights_only=True)
        if emb["embeddings"].shape[0] != data.x.shape[0]:
            raise ValueError(f"{pid}: {emb['embeddings'].shape[0]} embeddings for {data.x.shape[0]} residues")
        if self.emb_dim is not None and emb["embeddings"].shape[1] != self.emb_dim:
            raise ValueError(f"{pid}: embeddings of width {emb['embeddings'].shape[1]} in {self.emb_dir}, "
                             f"the model takes {self.emb_dim}")
        data.pair_id = pid
        data.seq_emb = emb["embeddings"]
        data.global_token = emb["global_token"].unsqueeze(0)
        return data
