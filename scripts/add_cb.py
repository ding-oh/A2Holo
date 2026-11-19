"""Add Cβ to SBAlign's Cα-only D3PM pairs, in the frame of the stored Cα.

    python scripts/add_cb.py --pt_dir <SBAlign Cα-only dir> --cif_dir <conformations> --output_dir data/d3pm/pairs

Each pair's residues are re-read from the apo and holo CIFs with SBAlign's own helpers, apo Cβ gets
the Kabsch transform that superposes apo Cα on holo Cα, and each structure is centred on its Cα
centroid. Adds pos_0_cb, pos_T_cb [N, 3] and cb_mask [N] (1 only where both structures have an
observed Cβ; glycine gets AlphaFold2's virtual Cβ).
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# AlphaFold2 residue_constants: CB = A * (b x c) + B * b + C * c + CA, b = CA - N, c = C - CA
VIRTUAL_CB_A, VIRTUAL_CB_B, VIRTUAL_CB_C = -0.58273431, 0.56802827, -0.54067466


def atom_coords(residue_df, name):
    """One atom's coordinates, averaged over alternate locations; None if the atom is absent."""
    rows = residue_df[residue_df["label_atom_id"] == name]
    if len(rows) == 0:
        return None
    coords = rows[["Cartn_x", "Cartn_y", "Cartn_z"]].values
    if coords.shape[0] > 1:
        coords = coords.mean(axis=0, keepdims=True)
    return coords.squeeze()


def residue_cb(residues, gather_ca_coords):
    """Cβ [N, 3], Cβ mask [N] and Cα [N, 3] of a residue dict."""
    n = len(residues)
    cb, mask, ca = np.zeros((n, 3)), np.zeros(n), np.zeros((n, 3))
    for i, (residue_id, df) in enumerate(residues.items()):
        ca[i] = gather_ca_coords(df)
        real = atom_coords(df, "CB")
        if real is not None:
            cb[i], mask[i] = real, 1.0
            continue
        n_xyz, ca_xyz, c_xyz = atom_coords(df, "N"), atom_coords(df, "CA"), atom_coords(df, "C")
        if n_xyz is not None and ca_xyz is not None and c_xyz is not None:
            b, c = ca_xyz - n_xyz, c_xyz - ca_xyz
            cb[i] = VIRTUAL_CB_A * np.cross(b, c) + VIRTUAL_CB_B * b + VIRTUAL_CB_C * c + ca_xyz
        else:
            warnings.warn(f"Missing backbone atoms for residue {residue_id}, using CA for CB")
            cb[i] = ca[i]
    return cb, mask, ca


def process_pair(pair_id, pt_path, cif_dir, sbalign):
    """Returns (data with the Cβ fields, None) or (None, error message)."""
    parts = pair_id.split("_")
    if len(parts) != 4:
        return None, f"Invalid pair_id format: {pair_id}"
    apo_pdb, apo_chain, holo_pdb, holo_chain = parts
    data = torch.load(pt_path, weights_only=False)
    apo_cif, holo_cif = os.path.join(cif_dir, f"{apo_pdb}.cif"), os.path.join(cif_dir, f"{holo_pdb}.cif")
    for kind, cif in (("Apo", apo_cif), ("Holo", holo_cif)):
        if not os.path.exists(cif):
            return None, f"{kind} CIF not found: {cif}"
    try:
        struct_apo = sbalign.parse_biopandas(apo_cif, chain_id=apo_chain)
        struct_holo = sbalign.parse_biopandas(holo_cif, chain_id=holo_chain)
    except Exception as e:
        return None, f"CIF parse error: {e}"

    residues_apo, residues_holo = sbalign.align_residue_dicts(sbalign.get_residue_dict(df=struct_apo),
                                                              sbalign.get_residue_dict(df=struct_holo))
    if len(residues_apo) != data.x.shape[0]:
        return None, f"Residue count mismatch: common={len(residues_apo)}, pt={data.x.shape[0]}"
    apo_cb, apo_mask, apo_ca = residue_cb(residues_apo, sbalign.gather_ca_coords)
    holo_cb, holo_mask, holo_ca = residue_cb(residues_holo, sbalign.gather_ca_coords)

    rot_mat, tr = sbalign.rigid_transform_kabsch_3D(apo_ca.T, holo_ca.T)
    apo_cb_aligned = (rot_mat @ apo_cb.T + tr).T
    apo_ca_aligned = (rot_mat @ apo_ca.T + tr).T
    apo_ca_com = np.mean(apo_ca_aligned, axis=0, keepdims=True)
    holo_ca_com = np.mean(holo_ca, axis=0, keepdims=True)

    # The re-derived Cα must reproduce the stored Cα, or the Cβ would sit in another frame.
    apo_diff = np.abs(apo_ca_aligned - apo_ca_com - data.pos_0.numpy()).max()
    holo_diff = np.abs(holo_ca - holo_ca_com - data.pos_T.numpy()).max()
    if apo_diff > 0.01:
        return None, f"Apo CA verification failed: max diff={apo_diff:.4f}Å"
    if holo_diff > 0.01:
        return None, f"Holo CA verification failed: max diff={holo_diff:.4f}Å"

    data.pos_0_cb = torch.tensor(apo_cb_aligned - apo_ca_com, dtype=torch.float32)
    data.pos_T_cb = torch.tensor(holo_cb - holo_ca_com, dtype=torch.float32)
    data.cb_mask = torch.tensor(np.minimum(apo_mask, holo_mask), dtype=torch.float32)
    return data, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt_dir", required=True, help="SBAlign's Cα-only <pair>.pt files")
    ap.add_argument("--cif_dir", required=True, help="D3PM conformations/ with <PDB id>.cif")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sbalign_dir", default=os.path.join(ROOT, "data/sbalign"),
                    help="SBAlign checkout providing proteins.conf.utils")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.sbalign_dir)
    from proteins.conf import utils as sbalign

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    done, skipped, errors = 0, 0, []
    for pt_file in tqdm(sorted(Path(args.pt_dir).glob("*.pt")), desc="adding Cβ"):
        out = output_dir / f"{pt_file.stem}.pt"
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        data, error = process_pair(pt_file.stem, str(pt_file), args.cif_dir, sbalign)
        if error:
            errors.append((pt_file.stem, error))
            continue
        torch.save(data, out)
        done += 1

    print(f"wrote {done}, skipped {skipped} existing, failed {len(errors)}")
    if errors:
        with open(output_dir / "extraction_errors.txt", "w") as f:
            f.writelines(f"{pid}\t{err}\n" for pid, err in errors)
        print(f"errors: {output_dir / 'extraction_errors.txt'}")


if __name__ == "__main__":
    main()
