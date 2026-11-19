"""The A2Holo network: a dual-stream transformer predicting Cα and Cβ velocities.

The structure stream carries the interpolated Cα+Cβ coordinates (Fourier-encoded) with node
features; the sequence stream carries frozen per-residue protein language model embeddings (Ankh3
in the reported model). Double-stream blocks keep separate weights per stream and couple them only
in joint attention over the concatenated tokens [sequence; structure]; single-stream blocks then
process the merged tokens. Both streams share their RoPE position ids, so residue i of either
stream sits at position i.
"""

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

NODE_DIM = 145    # node features from the SBAlign preprocessing
PLM_DIMS = dict(ankh3=1536, esm2=1280, esmc=960)   # Ankh3-large, ESM-2 650M and ESM-C 300M widths
COORD_DIM = 6     # Cα (3) + Cβ (3)


def rope_1d(pos: Tensor, dim: int, theta: int = 10000) -> Tensor:
    """1D rotary position embedding, computed in fp32 even inside an autocast region."""
    with torch.autocast(device_type=pos.device.type, enabled=False):
        scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
        omega = 1.0 / (theta ** scale)
        out = pos.float().unsqueeze(-1) * omega
        cos_out, sin_out = torch.cos(out), torch.sin(out)
        freqs = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
        freqs = rearrange(freqs, "b l d (i j) -> b l d i j", i=2, j=2)
    return freqs.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs.unsqueeze(1)
    xq_out = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_out = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return rearrange(x, "B H L D -> B L (H D)")


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None] * 1000.0
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    """AdaLN shift, scale and gate from the conditioning vector."""

    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim)

    def forward(self, cond: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(cond))[:, None, :].chunk(self.multiplier, dim=-1)
        return ModulationOut(*out[:3]), (ModulationOut(*out[3:]) if self.is_double else None)


class AttentionProjections(nn.Module):
    """Per-stream query/key/value projection, QK normalisation and output projection."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.norm = QKNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim)


def _mlp(hidden_size: int, mlp_ratio: float) -> nn.Sequential:
    hidden = int(hidden_size * mlp_ratio)
    return nn.Sequential(nn.Linear(hidden_size, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, hidden_size))


class DoubleStreamBlock(nn.Module):
    """Separate weights per stream; the two meet only in joint attention over [seq; struct]."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.num_heads = num_heads
        self.struct_mod = Modulation(hidden_size, double=True)
        self.struct_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.struct_attn = AttentionProjections(hidden_size, num_heads)
        self.struct_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.struct_mlp = _mlp(hidden_size, mlp_ratio)
        self.seq_mod = Modulation(hidden_size, double=True)
        self.seq_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.seq_attn = AttentionProjections(hidden_size, num_heads)
        self.seq_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.seq_mlp = _mlp(hidden_size, mlp_ratio)

    def forward(self, struct: Tensor, seq: Tensor, cond: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        struct_mod1, struct_mod2 = self.struct_mod(cond)
        seq_mod1, seq_mod2 = self.seq_mod(cond)

        struct_m = (1 + struct_mod1.scale) * self.struct_norm1(struct) + struct_mod1.shift
        struct_q, struct_k, struct_v = rearrange(self.struct_attn.qkv(struct_m), "B L (K H D) -> K B H L D",
                                                 K=3, H=self.num_heads)
        struct_q, struct_k = self.struct_attn.norm(struct_q, struct_k, struct_v)

        seq_m = (1 + seq_mod1.scale) * self.seq_norm1(seq) + seq_mod1.shift
        seq_q, seq_k, seq_v = rearrange(self.seq_attn.qkv(seq_m), "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        seq_q, seq_k = self.seq_attn.norm(seq_q, seq_k, seq_v)

        q = torch.cat((seq_q, struct_q), dim=2)
        k = torch.cat((seq_k, struct_k), dim=2)
        v = torch.cat((seq_v, struct_v), dim=2)
        attn = attention(q, k, v, pe=pe)
        seq_out, struct_out = attn[:, :seq.shape[1]], attn[:, seq.shape[1]:]

        struct = struct + struct_mod1.gate * self.struct_attn.proj(struct_out)
        struct = struct + struct_mod2.gate * self.struct_mlp(
            (1 + struct_mod2.scale) * self.struct_norm2(struct) + struct_mod2.shift)
        seq = seq + seq_mod1.gate * self.seq_attn.proj(seq_out)
        seq = seq + seq_mod2.gate * self.seq_mlp((1 + seq_mod2.scale) * self.seq_norm2(seq) + seq_mod2.shift)
        return struct, seq


class CrossAttentionDoubleStreamBlock(nn.Module):
    """Double-stream block with cross-attention instead of joint attention: each stream attends only
    to the other. Same parameter count as DoubleStreamBlock. struct_k_norm and seq_k_norm normalise
    the keys that the structure and sequence queries attend to."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.struct_mod = Modulation(hidden_size, double=True)
        self.struct_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.struct_q = nn.Linear(hidden_size, hidden_size)
        self.struct_kv = nn.Linear(hidden_size, hidden_size * 2)
        self.struct_q_norm = RMSNorm(head_dim)
        self.struct_k_norm = RMSNorm(head_dim)
        self.struct_proj = nn.Linear(hidden_size, hidden_size)
        self.struct_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.struct_mlp = _mlp(hidden_size, mlp_ratio)
        self.seq_mod = Modulation(hidden_size, double=True)
        self.seq_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.seq_q = nn.Linear(hidden_size, hidden_size)
        self.seq_kv = nn.Linear(hidden_size, hidden_size * 2)
        self.seq_q_norm = RMSNorm(head_dim)
        self.seq_k_norm = RMSNorm(head_dim)
        self.seq_proj = nn.Linear(hidden_size, hidden_size)
        self.seq_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.seq_mlp = _mlp(hidden_size, mlp_ratio)

    def forward(self, struct: Tensor, seq: Tensor, cond: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        H = self.num_heads
        pe_seq, pe_struct = pe[:, :seq.shape[1]], pe[:, seq.shape[1]:]
        struct_mod1, struct_mod2 = self.struct_mod(cond)
        seq_mod1, seq_mod2 = self.seq_mod(cond)
        struct_m = (1 + struct_mod1.scale) * self.struct_norm1(struct) + struct_mod1.shift
        seq_m = (1 + seq_mod1.scale) * self.seq_norm1(seq) + seq_mod1.shift

        struct_q = self.struct_q_norm(rearrange(self.struct_q(struct_m), "B L (H D) -> B H L D", H=H))
        seq_k, seq_v = rearrange(self.seq_kv(seq_m), "B L (K H D) -> K B H L D", K=2, H=H)
        seq_k = self.struct_k_norm(seq_k)
        seq_q = self.seq_q_norm(rearrange(self.seq_q(seq_m), "B L (H D) -> B H L D", H=H))
        struct_k, struct_v = rearrange(self.struct_kv(struct_m), "B L (K H D) -> K B H L D", K=2, H=H)
        struct_k = self.seq_k_norm(struct_k)

        struct_q, seq_k = apply_rope(struct_q, seq_k, pe_struct)
        seq_q, struct_k = apply_rope(seq_q, struct_k, pe_seq)
        sdpa = torch.nn.functional.scaled_dot_product_attention
        struct_out = rearrange(sdpa(struct_q, seq_k, seq_v), "B H L D -> B L (H D)")
        seq_out = rearrange(sdpa(seq_q, struct_k, struct_v), "B H L D -> B L (H D)")

        struct = struct + struct_mod1.gate * self.struct_proj(struct_out)
        struct = struct + struct_mod2.gate * self.struct_mlp(
            (1 + struct_mod2.scale) * self.struct_norm2(struct) + struct_mod2.shift)
        seq = seq + seq_mod1.gate * self.seq_proj(seq_out)
        seq = seq + seq_mod2.gate * self.seq_mlp((1 + seq_mod2.scale) * self.seq_norm2(seq) + seq_mod2.shift)
        return struct, seq


class SingleStreamBlock(nn.Module):
    """One set of weights over the merged tokens; attention and MLP run in parallel."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, cond: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(cond)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)
        return x + mod.gate * self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=-1))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        return self.linear((1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :])


class FourierFeatures(nn.Module):
    """Coordinates together with their sines and cosines at 10 octave frequencies."""

    def __init__(self, input_dim: int = COORD_DIM, num_frequencies: int = 10, coord_scale: float = 10.0):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(num_frequencies)) * math.pi / coord_scale)
        self.output_dim = input_dim * (2 * num_frequencies + 1)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.clamp(x, -100.0, 100.0)
        proj = x.unsqueeze(-1) * self.freqs
        return torch.cat([x, torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1).flatten(-2)], dim=-1)


class A2Holo(nn.Module):
    """Predicts (v_ca, v_cb), the Cα and Cβ velocities at flow time t.

    no_plm removes the sequence stream and the double-stream blocks; no_global_token zeroes the
    global token; node_features=False zeroes the node features at unchanged capacity.
    seq_from_structure is a control that keeps both streams and all blocks but feeds the sequence
    stream the structure stream's input features, through its own projection, instead of the PLM
    embedding; with no PLM token it needs no_global_token.
    attention="cross" uses CrossAttentionDoubleStreamBlock; plm selects the embeddings: "ankh3"
    (reported), "esm2" or "esmc".
    """

    def __init__(self, hidden_size: int = 768, num_heads: int = 8, depth_double: int = 12,
                 depth_single: int = 12, mlp_ratio: float = 4.0, no_plm: bool = False,
                 no_global_token: bool = False, node_features: bool = True,
                 seq_from_structure: bool = False, attention: str = "joint", plm: str = "ankh3"):
        super().__init__()
        if seq_from_structure and (no_plm or not no_global_token):
            raise ValueError("seq_from_structure needs the sequence stream (no no_plm) and no_global_token")
        if attention not in ("joint", "cross"):
            raise ValueError(f"attention {attention!r}: joint or cross")
        if attention == "cross" and no_plm:
            raise ValueError("cross attention needs the sequence stream (no no_plm)")
        if plm not in PLM_DIMS:
            raise ValueError(f"plm {plm!r}: one of {sorted(PLM_DIMS)}")
        self.config = dict(hidden_size=hidden_size, num_heads=num_heads, depth_double=depth_double,
                           depth_single=depth_single, mlp_ratio=mlp_ratio, no_plm=no_plm,
                           no_global_token=no_global_token, node_features=node_features,
                           seq_from_structure=seq_from_structure, attention=attention, plm=plm)
        self.plm_dim = PLM_DIMS[plm]
        self.no_plm = no_plm
        self.no_global_token = no_global_token
        self.node_features = node_features
        self.seq_from_structure = seq_from_structure
        self.head_dim = hidden_size // num_heads

        self.node_proj = nn.Linear(NODE_DIM, 72)
        self.coord_encoder = FourierFeatures()
        struct_dim = self.coord_encoder.output_dim + 72
        self.struct_in = nn.Linear(struct_dim, hidden_size)
        self.seq_in = nn.Linear(struct_dim if seq_from_structure else self.plm_dim, hidden_size)
        self.time_in = MLPEmbedder(256, hidden_size)
        self.global_in = MLPEmbedder(self.plm_dim, hidden_size)
        self.cond_norm = nn.LayerNorm(hidden_size * 2, elementwise_affine=False, eps=1e-6)
        self.cond_proj = nn.Linear(hidden_size * 2, hidden_size)
        double_block = CrossAttentionDoubleStreamBlock if attention == "cross" else DoubleStreamBlock
        self.double_blocks = nn.ModuleList(
            [double_block(hidden_size, num_heads, mlp_ratio) for _ in range(depth_double)])
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth_single)])
        self.final_layer = FinalLayer(hidden_size, COORD_DIM)

        # Xavier initialisation everywhere except node_proj, which keeps PyTorch's default;
        # the output layer starts at zero.
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and name != "node_proj":
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(self, x: Tensor, pos_ca: Tensor, pos_cb: Tensor, t: Tensor, batch_index: Tensor,
                seq_emb: Tensor, global_token: Tensor) -> tuple[Tensor, Tensor]:
        """x [N, 145] node features, pos_* [N, 3], t [B], batch_index [N], seq_emb [N, D] and
        global_token [B, D], D = PLM_DIMS[plm]. Returns v_ca [N, 3] and v_cb [N, 3]."""
        device = pos_ca.device
        B = int(batch_index.max().item()) + 1
        if not self.node_features:
            x = torch.zeros_like(x)
        node = self.node_proj(x)
        if self.no_global_token or self.no_plm:
            global_token = torch.zeros(B, self.plm_dim, device=device)

        cond = torch.cat([self.time_in(timestep_embedding(t, 256)), self.global_in(global_token)], dim=-1)
        cond = self.cond_proj(self.cond_norm(cond))
        coord = self.coord_encoder(torch.cat([pos_ca, pos_cb], dim=-1))

        # One protein at a time, so attention never crosses proteins in a batch.
        velocities = []
        for g in range(B):
            mask = batch_index == g
            L = int(mask.sum().item())
            cond_g = cond[g:g + 1]
            struct_input = torch.cat([coord[mask], node[mask]], dim=-1).unsqueeze(0)
            struct = self.struct_in(struct_input)
            if self.no_plm:
                pe = rope_1d(torch.arange(L, device=device).unsqueeze(0), self.head_dim)
                for block in self.single_blocks:
                    struct = block(struct, cond=cond_g, pe=pe)
            else:
                seq = self.seq_in(struct_input if self.seq_from_structure else seq_emb[mask].unsqueeze(0))
                ids = torch.arange(L, device=device).unsqueeze(0)
                pe = rope_1d(torch.cat([ids, ids], dim=1), self.head_dim)
                for block in self.double_blocks:
                    struct, seq = block(struct=struct, seq=seq, cond=cond_g, pe=pe)
                h = torch.cat([seq, struct], dim=1)
                for block in self.single_blocks:
                    h = block(h, cond=cond_g, pe=pe)
                struct = h[:, L:]
            velocities.append(self.final_layer(struct, cond_g)[0])
        v = torch.cat(velocities, dim=0)
        return v[:, :3], v[:, 3:]
