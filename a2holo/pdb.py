"""Writing Cα / Cβ traces as PDB files."""

import os

import numpy as np

# Residue order of the one-hot block x[:, :20] of the node features.
AA_1 = "ACDEFGHIKLMNPQRSTVWY"
AA_3 = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
        "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]


def sequence(x):
    """One-letter sequence from the node features x [N, >=20]."""
    return "".join(AA_1[i] for i in x[:, :20].argmax(dim=1).tolist())


def write_pdb(path, ca, seq, cb=None, cb_mask=None, bfactors=None):
    """Write Cα, and Cβ where cb_mask is set, as chain A numbered from 1."""
    ca = np.asarray(ca, dtype=float)
    bfactors = np.zeros(len(ca)) if bfactors is None else bfactors
    lines, serial = [], 1
    for i, (aa, b) in enumerate(zip(seq, bfactors)):
        atoms = [("CA", ca[i])] + ([("CB", cb[i])] if cb is not None and cb_mask[i] > 0.5 else [])
        for name, (x, y, z) in atoms:
            lines.append(f"ATOM  {serial:>5d}  {name:<2s}  {AA_3[AA_1.index(aa)]:>3s} A{i + 1:>4d}    "
                         f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{float(b):>6.2f}           C")
            serial += 1
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines + ["END"]) + "\n")
