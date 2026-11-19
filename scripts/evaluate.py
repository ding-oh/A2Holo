"""Evaluate a checkpoint: integrate the flow from each apo structure and score the prediction.

    python scripts/evaluate.py --checkpoint checkpoints/a2holo_seed1/best_model.pt

Writes a results JSON next to the checkpoint (or to --output) with the Cα and Cβ RMSD statistics,
the Cα lDDT and the per-pair values; --pdb_dir also writes pred/, target/ and apo/ PDB files for
TM-scoring. A checkpoint trained on another split is scored with that split's file (--splits).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2holo.checkpoint import load_model  # noqa: E402
from a2holo.data import DATA_DIR, SPLITS, PairDataset, emb_dir_for, split_ids  # noqa: E402
from a2holo.geometry import ca_lddt, kabsch, superposed_rmsd  # noqa: E402
from a2holo.pdb import sequence, write_pdb  # noqa: E402


@torch.no_grad()
def integrate(model, batch, num_steps):
    """Euler integration from the apo structure at t = 0 to t = 1."""
    pos_ca, pos_cb = batch.pos_0.clone(), batch.pos_0_cb.clone()
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((batch.num_graphs,), step * dt, device=pos_ca.device)
        v_ca, v_cb = model(batch.x, pos_ca, pos_cb, t, batch.batch, batch.seq_emb, batch.global_token)
        pos_ca, pos_cb = pos_ca + dt * v_ca, pos_cb + dt * v_cb
    return pos_ca, pos_cb


def write_pair_pdbs(pdb_dir, batch, pred_ca, pred_cb):
    """Prediction superposed on holo (Cβ under Cα's transform, B-factor = per-residue Cα error),
    and the holo and apo structures as given."""
    aligned_ca, R, cP, cQ = kabsch(pred_ca, batch.pos_T)
    aligned_cb = (pred_cb.float() - cP) @ R + cQ
    err = torch.linalg.norm(aligned_ca - batch.pos_T, dim=-1)
    seq, mask = sequence(batch.x), batch.cb_mask.cpu().numpy()
    pid = batch.pair_id[0]
    for name, ca, cb, b in [("pred", aligned_ca, aligned_cb, err), ("target", batch.pos_T, batch.pos_T_cb, None),
                            ("apo", batch.pos_0, batch.pos_0_cb, None)]:
        write_pdb(os.path.join(pdb_dir, name, f"{pid}.pdb"), ca.cpu().numpy(), seq, cb.cpu().numpy(), mask,
                  None if b is None else b.cpu().numpy())


def summarize(values):
    v = np.asarray([x for x in values if not np.isnan(x)])
    return dict(median=float(np.median(v)), mean=float(v.mean()),
                **{f"pct_below_{c}A": float((v < c).mean() * 100) for c in (2, 5, 10)}, n=int(v.size))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_steps", type=int, default=10, help="Euler steps (the reported results use 10)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--emb_dir", help="default: $EMB_DIR, else data/d3pm/<the checkpoint's language model>")
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--output", help="results JSON (default: <checkpoint>_<split>_results.json)")
    ap.add_argument("--pdb_dir", help="also write pred/, target/ and apo/ PDB files here")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    emb_dir = args.emb_dir or emb_dir_for(model.config["plm"])
    emb_dim = None if (model.no_plm or model.seq_from_structure) else model.plm_dim
    loader = DataLoader(PairDataset(args.data_dir, emb_dir, split_ids(args.splits, args.split), emb_dim),
                        batch_size=1)

    pair_ids, ca, cb, apo, lddt, apo_lddt = [], [], [], [], [], []
    for batch in tqdm(loader, desc=f"{args.split}, {args.num_steps} steps"):
        batch = batch.to(device)
        pred_ca, pred_cb = integrate(model, batch, args.num_steps)
        r_ca, r_cb = superposed_rmsd(pred_ca, pred_cb, batch.pos_T, batch.pos_T_cb, batch.cb_mask)
        pair_ids.append(batch.pair_id[0])
        ca.append(r_ca)
        cb.append(r_cb)
        apo.append(superposed_rmsd(batch.pos_0, batch.pos_0_cb, batch.pos_T, batch.pos_T_cb, batch.cb_mask)[0])
        lddt.append(ca_lddt(pred_ca.cpu(), batch.pos_T.cpu()))
        apo_lddt.append(ca_lddt(batch.pos_0.cpu(), batch.pos_T.cpu()))
        if args.pdb_dir:
            write_pair_pdbs(args.pdb_dir, batch, pred_ca, pred_cb)

    results = dict(checkpoint=args.checkpoint, epoch=ckpt.get("epoch"), split=args.split,
                   num_steps=args.num_steps, ca=summarize(ca), cb=summarize(cb), apo_ca=summarize(apo),
                   ca_lddt=dict(median=float(np.median(lddt)), mean=float(np.mean(lddt))),
                   apo_ca_lddt=dict(median=float(np.median(apo_lddt)), mean=float(np.mean(apo_lddt))),
                   pair_ids=pair_ids, ca_rmsd=ca, cb_rmsd=cb, apo_ca_rmsd=apo,
                   ca_lddt_per_pair=lddt, apo_ca_lddt_per_pair=apo_lddt)
    for key, label in [("ca", "Cα"), ("cb", "Cβ (glycine excluded)"), ("apo_ca", "apo input Cα")]:
        s = results[key]
        print(f"{label:>22}: median {s['median']:.3f} Å, mean {s['mean']:.3f} Å, "
              f"<2 Å {s['pct_below_2A']:.1f} %, <5 Å {s['pct_below_5A']:.1f} %  (n = {s['n']})")
    print(f"{'Cα lDDT':>22}: mean {results['ca_lddt']['mean']:.3f}, median {results['ca_lddt']['median']:.3f}  "
          f"(apo input: mean {results['apo_ca_lddt']['mean']:.3f})")
    out = args.output or args.checkpoint.replace(".pt", f"_{args.split}_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
