"""Kabsch superposition, the superposed RMSDs that every result is scored with, and Cα lDDT."""

import numpy as np
import torch


def kabsch(P, Q):
    """Superpose P onto Q ([N, 3] each), in fp32 even inside an autocast region.

    Returns (P_aligned, R, cP, cQ) with P_aligned = (P - cP) @ R + cQ, so the same transform can
    be applied to other points of the same structure (Cβ reuses the Cα superposition).
    """
    with torch.amp.autocast("cuda", enabled=False):
        P, Q = P.float(), Q.float()
        cP, cQ = P.mean(0, keepdim=True), Q.mean(0, keepdim=True)
        P0, Q0 = P - cP, Q - cQ
        U, _, Vt = torch.linalg.svd(P0.T @ Q0)
        d = torch.sign(torch.linalg.det(Vt.T @ U.T)).detach().reshape(1)
        R = U @ torch.diag(torch.cat([torch.ones(2, device=P.device, dtype=P.dtype), d])) @ Vt
        return P0 @ R + cQ, R, cP, cQ


@torch.no_grad()
def superposed_rmsd(pred_ca, pred_cb, true_ca, true_cb, cb_mask):
    """Cα RMSD after Kabsch superposition, and Cβ RMSD under the same transform over residues
    with an observed Cβ (NaN when there is none). One protein; returns two floats."""
    aligned_ca, R, cP, cQ = kabsch(pred_ca, true_ca)
    ca = torch.sqrt(((aligned_ca - true_ca) ** 2).sum(dim=-1).mean()).item()
    m = cb_mask.bool()
    if not m.any():
        return ca, float("nan")
    with torch.amp.autocast("cuda", enabled=False):
        aligned_cb = (pred_cb.float() - cP) @ R + cQ
    cb = torch.sqrt(((aligned_cb[m] - true_cb.float()[m]) ** 2).sum(dim=-1).mean()).item()
    return ca, cb


def ca_lddt(pred_ca, true_ca, cutoff=15.0):
    """Cα lDDT of a prediction against the holo structure, alignment-free: for each residue, the
    fraction of its distances to residues within `cutoff` in holo that are preserved to 0.5, 1, 2
    and 4 Å (averaged over the four thresholds), then the mean over residues."""
    p = np.asarray(pred_ca, dtype=np.float64)
    t = np.asarray(true_ca, dtype=np.float64)
    d_pred = np.linalg.norm(p[:, None] - p[None], axis=-1)
    d_true = np.linalg.norm(t[:, None] - t[None], axis=-1)
    pairs = (d_true < cutoff) & ~np.eye(len(t), dtype=bool)
    diff = np.abs(d_pred - d_true)
    kept = np.mean([diff < th for th in (0.5, 1.0, 2.0, 4.0)], axis=0)
    return float(((kept * pairs).sum(1) / np.maximum(pairs.sum(1), 1)).mean())


def random_rotation(device):
    """A uniformly random proper rotation matrix."""
    Q, R = torch.linalg.qr(torch.randn(3, 3, device=device))
    Q = Q * torch.sign(torch.diag(R))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q
