"""Train A2Holo with the reported recipe; every default is the recipe.

    python scripts/train.py --seed 1

Flow matching between the apo and holo structures, x_t = (1 - t) x_apo + t x_holo with t ~ Beta(2, 2),
under a random rotation per protein. The model predicts the velocity v, the prediction x_apo + v is
scored, and the loss is

    L = w_ca D_Cα + w_cb D_Cβ + w_slddt (1 - sLDDT_Cα) + w_rel L_rel

with D the mean per-atom distance after one Kabsch superposition on Cα (Cβ reuses it). Validation
integrates one Euler step with the EMA weights; the checkpoint with the lowest mean validation Cα
RMSD is kept as <out_dir>/best_model.pt. Training stops after 30 validations without improvement or
at epoch 700; for every seed and ablation arm in the paper this rule selects the reported checkpoint.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2holo.checkpoint import save_checkpoint  # noqa: E402
from a2holo.data import DATA_DIR, SPLITS, PairDataset, emb_dir_for, split_ids  # noqa: E402
from a2holo.geometry import kabsch, random_rotation  # noqa: E402
from a2holo.model import PLM_DIMS, A2Holo  # noqa: E402


def rotate_pairs(pos_0, pos_T, pos_0_cb, pos_T_cb, batch_index):
    """One random rotation per protein about its apo centroid, applied to all four coordinate sets."""
    out = [p.clone() for p in (pos_0, pos_T, pos_0_cb, pos_T_cb)]
    for g in range(int(batch_index.max().item()) + 1):
        mask = batch_index == g
        R = random_rotation(pos_0.device)
        center = pos_0[mask].mean(dim=0, keepdim=True)
        for src, dst in zip((pos_0, pos_T, pos_0_cb, pos_T_cb), out):
            dst[mask] = (src[mask] - center) @ R.T + center
    return out


def kabsch_mae(v_ca, v_cb, pos_0, pos_T, pos_0_cb, pos_T_cb, cb_mask, batch_index):
    """Mean per-atom distance of x_apo + v to holo after Kabsch superposition on Cα; Cβ reuses the
    Cα transform and is averaged over residues with an observed Cβ. Returns (D_Cα, D_Cβ)."""
    ca_losses, cb_losses = [], []
    for g in range(int(batch_index.max().item()) + 1):
        mask = batch_index == g
        aligned_ca, R, cP, cQ = kabsch(pos_0[mask] + v_ca[mask], pos_T[mask])
        ca_losses.append(torch.linalg.norm(aligned_ca - pos_T[mask], dim=-1).mean())
        with torch.amp.autocast("cuda", enabled=False):
            aligned_cb = ((pos_0_cb[mask] + v_cb[mask]).float() - cP.float()) @ R.float() + cQ.float()
        dist_cb = torch.linalg.norm(aligned_cb - pos_T_cb[mask].float(), dim=-1)
        m = cb_mask[mask]
        if m.sum() > 0:
            cb_losses.append((dist_cb * m).sum() / m.sum())
    ca_loss = torch.stack(ca_losses).mean()
    cb_loss = torch.stack(cb_losses).mean() if cb_losses else torch.zeros((), device=ca_loss.device)
    return ca_loss, cb_loss


def smooth_lddt_loss(pred_ca, true_ca, batch_index, cutoff=15.0):
    """1 - smooth lDDT on Cα: sigmoids at the 0.5, 1, 2 and 4 Å thresholds over pairs closer than
    `cutoff` in the holo structure. Alignment-free."""
    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=pred_ca.device)
    losses = []
    for g in range(int(batch_index.max().item()) + 1):
        mask = batch_index == g
        pred, true = pred_ca[mask].float(), true_ca[mask].float()
        L = pred.shape[0]
        d_pred = torch.cdist(pred.unsqueeze(0), pred.unsqueeze(0)).squeeze(0)
        d_true = torch.cdist(true.unsqueeze(0), true.unsqueeze(0)).squeeze(0)
        pairs = (d_true < cutoff) & ~torch.eye(L, device=pred.device, dtype=torch.bool)
        if pairs.sum() == 0:
            continue
        diff = torch.abs(d_pred[pairs] - d_true[pairs])
        losses.append(1.0 - torch.sigmoid(thresholds[:, None] - diff[None, :]).mean(dim=0).mean())
    return torch.stack(losses).mean() if losses else torch.zeros((), device=pred_ca.device)


def relative_displacement_loss(v_ca, pos_0, pos_T, batch_index, k=16, seq_window=2):
    """Mean over neighbour pairs (i, j) of ||(v_i - v_j) - (v*_i - v*_j)||, v* = x_holo - x_apo.

    Neighbours are the k nearest residues in the apo structure plus `seq_window` sequence
    neighbours on each side, so a field that is right on average but locally incoherent is
    penalised. fp32 throughout: bf16 would perturb both the neighbour choice and the differences.
    """
    losses = []
    with torch.amp.autocast("cuda", enabled=False):
        for g in range(int(batch_index.max().item()) + 1):
            mask = batch_index == g
            p0 = pos_0[mask].float()
            L = p0.shape[0]
            with torch.no_grad():
                d = torch.cdist(p0, p0)
                d.fill_diagonal_(float("inf"))
                nbr = d.topk(min(k, L - 1), dim=-1, largest=False).indices
                ar = torch.arange(L, device=p0.device)
                offsets = [o for o in range(-seq_window, seq_window + 1) if o != 0]
                nbr = torch.cat([nbr, torch.stack([(ar + o).clamp(0, L - 1) for o in offsets], dim=-1)], dim=-1)
            v = v_ca[mask].float()
            v_true = (pos_T[mask] - pos_0[mask]).float()
            losses.append(((v.unsqueeze(1) - v[nbr]) - (v_true.unsqueeze(1) - v_true[nbr])).norm(dim=-1).mean())
    return torch.stack(losses).mean()


class EMA:
    """Exponential moving average of the parameters, with the usual warm-up of the decay."""

    def __init__(self, params, decay):
        self.decay, self.num_updates = decay, 0
        self.shadow = [p.clone().detach() for p in params]

    def update(self, params):
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        self.num_updates += 1
        with torch.no_grad():
            for s, p in zip(self.shadow, params):
                s.mul_(decay).add_(p.data, alpha=1.0 - decay)

    def state_dict(self, model):
        """The model's state dict with the EMA in place of each parameter."""
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for (name, _), s in zip(model.named_parameters(), self.shadow):
            sd[name] = s.detach().clone()
        return sd


def train_epoch(model, loader, optimizer, ema, device, args):
    model.train()
    sums, n_steps, n_skipped = np.zeros(5), 0, 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        idx, B = batch.batch, batch.num_graphs
        pos_0, pos_T, pos_0_cb, pos_T_cb = rotate_pairs(batch.pos_0, batch.pos_T, batch.pos_0_cb, batch.pos_T_cb, idx)
        t = torch.distributions.Beta(2.0, 2.0).sample((B,)).to(device)
        t_node = t[idx].unsqueeze(-1)
        pos_t_ca = (1 - t_node) * pos_0 + t_node * pos_T
        pos_t_cb = (1 - t_node) * pos_0_cb + t_node * pos_T_cb

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_ca, v_cb = model(batch.x, pos_t_ca, pos_t_cb, t, idx, batch.seq_emb, batch.global_token)
            ca_loss, cb_loss = kabsch_mae(v_ca, v_cb, pos_0, pos_T, pos_0_cb, pos_T_cb, batch.cb_mask, idx)
            loss = args.w_ca * ca_loss + args.w_cb * cb_loss
            slddt = rel = torch.zeros((), device=device)
            if args.w_slddt > 0:
                slddt = smooth_lddt_loss(pos_0 + v_ca, pos_T, idx)
                loss = loss + args.w_slddt * slddt
            if args.w_rel > 0:
                rel = relative_displacement_loss(v_ca, pos_0, pos_T, idx)
                loss = loss + args.w_rel * rel

        # A non-finite loss or gradient would write NaN into every weight and the EMA; skip the step.
        if not torch.isfinite(loss):
            n_skipped += 1
            continue
        loss.backward()
        if not torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)):
            optimizer.zero_grad(set_to_none=True)
            n_skipped += 1
            continue
        optimizer.step()
        ema.update(model.parameters())
        sums += [loss.item(), ca_loss.item(), cb_loss.item(), slddt.item(), rel.item()]
        n_steps += 1
    return dict(zip(["loss", "ca", "cb", "slddt", "rel"], sums / max(n_steps, 1))), n_skipped


@torch.inference_mode()
def validate(model, loader, device, num_steps):
    """Mean Cα RMSD over the validation pairs after `num_steps` Euler steps from apo."""
    model.eval()
    rmsds = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        pos_ca, pos_cb = batch.pos_0.clone(), batch.pos_0_cb.clone()
        dt = 1.0 / num_steps
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for step in range(num_steps):
                t = torch.full((batch.num_graphs,), step * dt, device=device)
                v_ca, v_cb = model(batch.x, pos_ca, pos_cb, t, batch.batch, batch.seq_emb, batch.global_token)
                pos_ca, pos_cb = pos_ca + dt * v_ca, pos_cb + dt * v_cb
        for g in range(batch.num_graphs):
            mask = batch.batch == g
            aligned = kabsch(pos_ca[mask], batch.pos_T[mask])[0]
            rmsds.append(torch.sqrt(((aligned - batch.pos_T[mask]) ** 2).sum(dim=-1).mean()).item())
    return float(np.mean(np.asarray(rmsds, dtype=np.float64)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", help="default: checkpoints/a2holo_seed<seed>")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--emb_dir", help="default: $EMB_DIR, else data/d3pm/<plm>")
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate at batch size 32, scaled linearly")
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=700, help="last epoch, validated if on the schedule")
    ap.add_argument("--val_freq", type=int, default=5, help="validate every this many epochs")
    ap.add_argument("--patience", type=int, default=30, help="validations without improvement before stopping")
    ap.add_argument("--val_num_steps", type=int, default=1)
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    ap.add_argument("--w_ca", type=float, default=0.5)
    ap.add_argument("--w_cb", type=float, default=0.5)
    ap.add_argument("--w_slddt", type=float, default=0.5)
    ap.add_argument("--w_rel", type=float, default=2.0)
    ap.add_argument("--hidden_size", type=int, default=768)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--depth_double", type=int, default=12)
    ap.add_argument("--depth_single", type=int, default=12)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--no_plm", action="store_true", help="ablation: no sequence stream (with --depth_double 0)")
    ap.add_argument("--no_global_token", action="store_true", help="ablation: zero the global token")
    ap.add_argument("--no_node_features", action="store_true", help="ablation: zero the node features")
    ap.add_argument("--seq_from_structure", action="store_true",
                    help="control: the sequence stream takes the structure stream's input features instead of "
                         "the PLM embedding (needs --no_global_token)")
    ap.add_argument("--attention", choices=("joint", "cross"), default="joint",
                    help="attention of the double-stream blocks; cross: each stream attends only to the other")
    ap.add_argument("--plm", choices=sorted(PLM_DIMS), default="ankh3",
                    help="language model of the embeddings and the global token (Ankh3 is the reported model)")
    args = ap.parse_args()
    if args.seq_from_structure and (args.no_plm or not args.no_global_token):
        ap.error("--seq_from_structure needs --no_global_token and excludes --no_plm")
    if args.attention == "cross" and args.no_plm:
        ap.error("--attention cross needs the sequence stream and excludes --no_plm")
    if args.plm != "ankh3" and (args.no_plm or args.seq_from_structure):
        ap.error("--plm applies only to models that take the embedding (no --no_plm, --seq_from_structure)")
    args.emb_dir = args.emb_dir or emb_dir_for(args.plm)
    out_dir = args.out_dir or f"checkpoints/a2holo_seed{args.seed}"
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    emb_dim = None if (args.no_plm or args.seq_from_structure) else PLM_DIMS[args.plm]
    train_set = PairDataset(args.data_dir, args.emb_dir, split_ids(args.splits, "train"), emb_dim)
    val_set = PairDataset(args.data_dir, args.emb_dir, split_ids(args.splits, "val"), emb_dim)
    model = A2Holo(hidden_size=args.hidden_size, num_heads=args.num_heads, depth_double=args.depth_double,
                   depth_single=args.depth_single, mlp_ratio=args.mlp_ratio, no_plm=args.no_plm,
                   no_global_token=args.no_global_token, node_features=not args.no_node_features,
                   seq_from_structure=args.seq_from_structure, attention=args.attention,
                   plm=args.plm).to(device)
    lr = args.lr * (args.batch_size / 32)
    print(f"{len(train_set)} train / {len(val_set)} val pairs, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters, lr {lr:.2e}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=4, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3,
                                                           min_lr=1e-7)
    ema = EMA(model.parameters(), args.ema_decay)

    best, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(args.epochs + 1):
        losses, n_skipped = train_epoch(model, train_loader, optimizer, ema, device, args)
        if epoch % args.val_freq:
            continue
        raw = [p.detach().clone() for p in model.parameters()]
        with torch.no_grad():
            for p, s in zip(model.parameters(), ema.shadow):
                p.copy_(s)
        val = validate(model, val_loader, device, args.val_num_steps)
        with torch.no_grad():
            for p, r in zip(model.parameters(), raw):
                p.copy_(r)
        scheduler.step(val)
        print(f"epoch {epoch}: loss {losses['loss']:.4f} (Cα {losses['ca']:.4f}, Cβ {losses['cb']:.4f}, "
              f"sLDDT {losses['slddt']:.4f}, rel {losses['rel']:.4f}), val Cα RMSD {val:.3f} Å, "
              f"lr {optimizer.param_groups[0]['lr']:.2e}" + (f", {n_skipped} steps skipped" if n_skipped else ""),
              flush=True)
        if val < best:
            best, best_epoch, stale = val, epoch, 0
            save_checkpoint(os.path.join(out_dir, "best_model.pt"), model, ema.state_dict(model),
                            epoch=epoch, val_ca_rmsd=val, train_args=vars(args))
        else:
            stale += 1
            if stale >= args.patience:
                break
    print(f"best validation Cα RMSD {best:.3f} Å at epoch {best_epoch}")


if __name__ == "__main__":
    main()
