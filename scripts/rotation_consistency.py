"""Rotation consistency: rerun the test evaluation under Haar-uniform SO(3) rotations of the input.

    python scripts/rotation_consistency.py --checkpoint checkpoints/a2holo_seed1/best_model.pt

Each rotation turns the apo and holo Cα and Cβ about the pair's apo Cα centroid before integration.
Two things are measured, and they differ by two orders of magnitude:

  accuracy   how much the score moves. ca_deviation_mean is the signed mean over rotations and pairs
             of (rotated - unrotated) Cα RMSD to holo; ca_deviation_abs_median is the median of its
             magnitude.
  the prediction itself  turned back into the original frame and compared with the unrotated one:
             turned_back_raw_* as it stands, _centred_* after removing the centroid offset, and
             _shape_* after superposition. Nearly all of the raw difference is translation, which
             neither the loss nor any reported metric sees.

Writes <output_dir>/rotation_random_results.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2holo.checkpoint import load_model  # noqa: E402
from a2holo.data import DATA_DIR, SPLITS, PairDataset, emb_dir_for, split_ids  # noqa: E402
from a2holo.geometry import kabsch, superposed_rmsd  # noqa: E402
from scripts.evaluate import integrate  # noqa: E402


def random_rotations(seed, n):
    rng = np.random.RandomState(seed)
    return [torch.tensor(Rotation.random(random_state=rng).as_matrix(), dtype=torch.float32) for _ in range(n)]


def rotate(batch, R, centre):
    """Apo and holo Cα and Cβ turned by R about `centre`, on the CPU."""
    batch = batch.clone()
    for key in ("pos_0", "pos_0_cb", "pos_T", "pos_T_cb"):
        setattr(batch, key, (getattr(batch, key).float() - centre) @ R.T + centre)
    return batch


def rmsd(P, Q):
    return float(torch.sqrt(((P - Q) ** 2).sum(-1).mean()))


def moved(back, unrotated):
    """How far a turned-back prediction sits from the unrotated one: as it stands, after removing the
    centroid offset, and after superposition. Only the last is shape; nothing in the loss or in any
    reported metric sees the translation."""
    return (rmsd(back, unrotated),
            rmsd(back - back.mean(0), unrotated - unrotated.mean(0)),
            rmsd(kabsch(back, unrotated)[0], unrotated))


def predict(model, loader, device, num_steps, R=None, desc=""):
    """One pass over the split: per-pair Cα and Cβ RMSD to holo, the pair ids, and the predicted Cα
    in the original frame (turned back by R when the input was rotated)."""
    ca, cb, pair_ids, preds = [], [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        centre = batch.pos_0.float().mean(0, keepdim=True)
        if R is not None:
            batch = rotate(batch, R, centre)
        batch = batch.to(device)
        pred_ca, pred_cb = integrate(model, batch, num_steps)
        r_ca, r_cb = superposed_rmsd(pred_ca, pred_cb, batch.pos_T, batch.pos_T_cb, batch.cb_mask)
        ca.append(r_ca)
        cb.append(r_cb)
        pair_ids.append(batch.pair_id[0])
        p = pred_ca.float().cpu()
        preds.append(p if R is None else (p - centre) @ R + centre)
    return np.array(ca), np.array(cb), pair_ids, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_steps", type=int, default=10, help="Euler steps (the reported results use 10)")
    ap.add_argument("--num_rotations", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--emb_dir", help="default: $EMB_DIR, else data/d3pm/<the checkpoint's language model>")
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--output_dir", default="outputs/rotation_eval")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.checkpoint, device)
    emb_dir = args.emb_dir or emb_dir_for(model.config["plm"])
    emb_dim = None if (model.no_plm or model.seq_from_structure) else model.plm_dim
    loader = DataLoader(PairDataset(args.data_dir, emb_dir, split_ids(args.splits, "test"), emb_dim), batch_size=1)

    bl_ca, bl_cb, pair_ids, bl_pred = predict(model, loader, device, args.num_steps, desc="unrotated")
    rotations = random_rotations(args.seed, args.num_rotations)
    all_ca, all_cb, turned_back = [], [], []
    for k, R in enumerate(rotations):
        r_ca, r_cb, _, preds = predict(model, loader, device, args.num_steps, R,
                                       desc=f"rotation {k + 1}/{len(rotations)}")
        all_ca.append(r_ca)
        all_cb.append(r_cb)
        turned_back.append([moved(b, p) for b, p in zip(preds, bl_pred)])
    all_ca, all_cb = np.array(all_ca), np.array(all_cb)
    # [rotations, pairs, 3]: as it stands, centroid removed, superposed.
    turned_back = np.array(turned_back)

    n_prot = len(pair_ids)
    dev_ca = all_ca - bl_ca[None, :]
    dev_cb = all_cb - bl_cb[None, :]
    flat_ca, flat_cb = all_ca.flatten(), all_cb.flatten()
    pp_ca_mean, pp_ca_std = all_ca.mean(axis=0), all_ca.std(axis=0)
    # Pairs without an observed Cβ are NaN under every rotation and stay NaN here.
    cb_scored = ~np.isnan(bl_cb)
    pp_cb_mean, pp_cb_std = np.full(n_prot, np.nan), np.full(n_prot, np.nan)
    pp_cb_mean[cb_scored] = all_cb[:, cb_scored].mean(axis=0)
    pp_cb_std[cb_scored] = all_cb[:, cb_scored].std(axis=0)

    results = {
        "checkpoint": args.checkpoint,
        "num_steps": args.num_steps,
        "num_rotations": len(rotations),
        "seed": args.seed,
        "num_proteins": n_prot,
        "pair_ids": pair_ids,
        "baseline_ca_median": float(np.median(bl_ca)),
        "baseline_ca_mean": float(np.mean(bl_ca)),
        "baseline_ca_pct_under_2A": float((bl_ca < 2.0).mean() * 100),
        "baseline_ca_pct_under_5A": float((bl_ca < 5.0).mean() * 100),
        "baseline_cb_median": float(np.nanmedian(bl_cb)),
        "baseline_cb_mean": float(np.nanmean(bl_cb)),
        "cb_n_scored": int(cb_scored.sum()),
        "rotated_ca_median": float(np.median(flat_ca)),
        "rotated_ca_mean": float(np.mean(flat_ca)),
        "rotated_ca_pct_under_2A": float((flat_ca < 2.0).mean() * 100),
        "rotated_ca_pct_under_5A": float((flat_ca < 5.0).mean() * 100),
        "rotated_cb_median": float(np.nanmedian(flat_cb)),
        "rotated_cb_mean": float(np.nanmean(flat_cb)),
        "ca_deviation_mean": float(dev_ca.mean()),
        "ca_deviation_std": float(dev_ca.std()),
        "ca_deviation_abs_mean": float(np.abs(dev_ca).mean()),
        "ca_deviation_abs_median": float(np.median(np.abs(dev_ca))),
        "ca_deviation_abs_max": float(np.abs(dev_ca).max()),
        "pairs_with_abs_change_over_0p5A": int((np.abs(dev_ca).max(axis=0) > 0.5).sum()),
        "turned_back_raw_median": float(np.median(turned_back[:, :, 0])),
        "turned_back_raw_mean": float(turned_back[:, :, 0].mean()),
        "turned_back_centred_median": float(np.median(turned_back[:, :, 1])),
        "turned_back_shape_median": float(np.median(turned_back[:, :, 2])),
        "turned_back_shape_mean": float(turned_back[:, :, 2].mean()),
        "cb_deviation_mean": float(np.nanmean(dev_cb)),
        "cb_deviation_std": float(np.nanstd(dev_cb)),
        "per_protein": {
            pid: {
                "baseline_ca": float(bl_ca[i]),
                "baseline_cb": float(bl_cb[i]),
                "rotated_ca_mean": float(pp_ca_mean[i]),
                "rotated_ca_std": float(pp_ca_std[i]),
                "rotated_cb_mean": float(pp_cb_mean[i]),
                "rotated_cb_std": float(pp_cb_std[i]),
                "ca_deviation_mean": float(dev_ca[:, i].mean()),
                "ca_deviation_std": float(dev_ca[:, i].std()),
                "turned_back_shape_median": float(np.median(turned_back[:, i, 2])),
            }
            for i, pid in enumerate(pair_ids)
        },
        "baseline_ca_rmsds": bl_ca.tolist(),
        "baseline_cb_rmsds": bl_cb.tolist(),
        "all_rot_ca_rmsds": all_ca.tolist(),
        "all_rot_cb_rmsds": all_cb.tolist(),
    }

    print(f"{'':<16} {'unrotated':>10} {'rotated':>10}")
    for label, key, fmt in [("Cα median (Å)", "ca_median", ".4f"), ("Cα mean (Å)", "ca_mean", ".4f"),
                            ("Cα < 2 Å (%)", "ca_pct_under_2A", ".1f"), ("Cβ median (Å)", "cb_median", ".4f")]:
        print(f"{label:<16} {results['baseline_' + key]:>10{fmt}} {results['rotated_' + key]:>10{fmt}}")
    print(f"{len(rotations)} rotations x {n_prot} pairs")
    print(f"  Cα RMSD to holo changes by: signed mean {results['ca_deviation_mean']:+.4f} Å, "
          f"median |change| {results['ca_deviation_abs_median']:.4f}, |max| {results['ca_deviation_abs_max']:.4f} "
          f"({results['pairs_with_abs_change_over_0p5A']} of {n_prot} pairs move by more than 0.5 Å)")
    print(f"  the prediction itself moves (median): {results['turned_back_raw_median']:.4f} Å as it stands, "
          f"{results['turned_back_centred_median']:.4f} centred, {results['turned_back_shape_median']:.4f} in shape")
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "rotation_random_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
