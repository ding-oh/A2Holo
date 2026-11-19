"""Shared paths, loaders and attention capture for the analysis scripts.

run.py fills MODELS and T_VAL before a script starts, so no script names a checkpoint or a time itself.
"""
import math
import os
import sys

import torch
from einops import rearrange

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from a2holo import checkpoint  # noqa: E402
from a2holo.data import DATA_DIR, EMB_DIR, SPLITS, split_ids  # noqa: E402
from a2holo.model import apply_rope  # noqa: E402

OUT_DIR = os.environ.get("ANALYSIS_OUT", os.path.join(ROOT, "outputs/analysis"))
# (label, directory holding best_model.pt)
MODELS = []
# The analysed input is (1 - t) * apo + t * holo.
T_VAL = 0.5
# Bound on protein length, so the 2L x 2L attention maps fit in memory.
MAX_LEN = 600


def load_model(ckpt_path, device):
    """The checkpoint's model with its EMA weights, in eval mode; returns (model, depth_double)."""
    model, ckpt = checkpoint.load_model(ckpt_path, device)
    if model.config["attention"] != "joint" or model.config["plm"] != "ankh3":
        raise SystemExit(f"{ckpt_path}: {model.config['attention']} attention, {model.config['plm']} embeddings; "
                         "the analyses take joint attention and Ankh3 embeddings")
    return model, ckpt["config"]["depth_double"]


def test_pairs():
    """Test pairs that have both a structure file and an embedding file, in split order."""
    return [p for p in split_ids(SPLITS, "test")
            if os.path.exists(os.path.join(DATA_DIR, f"{p}.pt")) and os.path.exists(os.path.join(EMB_DIR, f"{p}.pt"))]


def load_pair(pid, device):
    """One pair's tensors on `device`, as a single-protein batch."""
    data = torch.load(os.path.join(DATA_DIR, f"{pid}.pt"), weights_only=False)
    emb = torch.load(os.path.join(EMB_DIR, f"{pid}.pt"), weights_only=True)
    n = data.x.shape[0]
    return dict(data=data, n=n, x=data.x.to(device), ca0=data.pos_0.to(device), caT=data.pos_T.to(device),
                cb0=data.pos_0_cb.to(device), cbT=data.pos_T_cb.to(device), cb_mask=data.cb_mask.to(device),
                bidx=torch.zeros(n, dtype=torch.long, device=device), seq_emb=emb["embeddings"].to(device),
                global_token=emb["global_token"].to(device).unsqueeze(0))


def kabsch_rotation(P, Q):
    """R minimising ||(P - cP) R - (Q - cQ)||, i.e. superposing P onto Q, in P's dtype."""
    Pc, Qc = P - P.mean(0, keepdim=True), Q - Q.mean(0, keepdim=True)
    U, _, Vt = torch.linalg.svd(Pc.T @ Qc)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d.item()], device=P.device, dtype=P.dtype))
    return U @ D @ Vt


def kabsch_rmsd(P, Q):
    """RMSD of P to Q after optimal superposition, in P's dtype."""
    Pc, Qc = P - P.mean(0, keepdim=True), Q - Q.mean(0, keepdim=True)
    return float(((Pc @ kabsch_rotation(P, Q) - Qc) ** 2).sum(-1).mean().sqrt())


def rho_perp(dH, H, eps=1e-12):
    """||dH_perp|| / ||dH||, dH_perp the part of each residue's update orthogonal to its state."""
    par = (dH * H).sum(-1, keepdim=True) / (H * H).sum(-1, keepdim=True).clamp_min(eps) * H
    return ((dH - par).norm() / dH.norm().clamp_min(eps)).item()


def cos_agg(a, b, eps=1e-12):
    """Cosine over all residues at once (one number per protein), not a per-residue mean."""
    return ((a * b).sum() / (a.norm() * b.norm()).clamp_min(eps)).item()


class AttentionCapturer:
    """Head-averaged joint attention of every double-stream block, rows and columns in storage order
    [sequence; structure].

    A forward hook recomputes each block's attention from the block's own inputs. The blocks are
    called with keyword arguments only, so the hook is registered with with_kwargs=True; call check()
    after a forward to confirm every block was captured.
    """

    def __init__(self, model):
        self.model = model
        self.captures = {}
        self._hooks = []

    def register(self):
        self.n_blocks = len(self.model.double_blocks)
        for i, block in enumerate(self.model.double_blocks):
            self._hooks.append(block.register_forward_hook(self._make_hook(i), with_kwargs=True))

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, kwargs, output):
            struct, seq, cond, pe = kwargs["struct"], kwargs["seq"], kwargs["cond"], kwargs["pe"]
            H = module.num_heads
            L = seq.shape[1]

            struct_mod1, _ = module.struct_mod(cond)
            seq_mod1, _ = module.seq_mod(cond)

            struct_m = (1 + struct_mod1.scale) * module.struct_norm1(struct) + struct_mod1.shift
            sq, sk, sv = rearrange(module.struct_attn.qkv(struct_m), "B L (K H D) -> K B H L D", K=3, H=H)
            sq, sk = module.struct_attn.norm(sq, sk, sv)

            seq_m = (1 + seq_mod1.scale) * module.seq_norm1(seq) + seq_mod1.shift
            qq, qk, qv = rearrange(module.seq_attn.qkv(seq_m), "B L (K H D) -> K B H L D", K=3, H=H)
            qq, qk = module.seq_attn.norm(qq, qk, qv)

            q = torch.cat((qq, sq), dim=2)
            k = torch.cat((qk, sk), dim=2)
            q, k = apply_rope(q, k, pe)
            attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
            self.captures[layer_idx] = {"attn": attn.mean(dim=1)[0].detach().cpu().numpy(), "L": L}

        return hook_fn

    def check(self):
        """Raise unless the last forward captured every double-stream block."""
        if len(self.captures) != self.n_blocks:
            raise RuntimeError(f"captured {len(self.captures)} of {self.n_blocks} double-stream blocks; "
                               "the attention hooks did not run")

    def clear(self):
        self.captures = {}

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
