"""Rebuild all-atom structures from Cα traces with cg2all (CalphaBasedModel).

    python scripts/allatom_cg2all.py --checkpoint checkpoints/a2holo_seed1/best_model.pt

Three tracks per test pair: pred (the predicted Cα), apo (lower bound) and holo (upper bound). Each
Cα trace is written to <output_dir>/ca_only_pdbs/<track>/<pair>.pdb and rebuilt by cg2all's
`convert_cg2all` ($CONVERT_CG2ALL, or on PATH; CPU) into <output_dir>/allatom_pdbs/<track>/<pair>.pdb.
manifest.json lists the files; all-atom files that already exist are kept.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2holo.checkpoint import load_model  # noqa: E402
from a2holo.data import DATA_DIR, EMB_DIR, SPLITS, PairDataset, split_ids  # noqa: E402
from a2holo.pdb import sequence, write_pdb  # noqa: E402
from scripts.evaluate import integrate  # noqa: E402

TRACKS = ("pred", "apo", "holo")


def write_ca_pdbs(args, tracks):
    """Predict each test pair and write the Cα-only PDBs of every track; returns the manifest."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.checkpoint, device)
    loader = DataLoader(PairDataset(args.data_dir, args.emb_dir, split_ids(args.splits, "test")), batch_size=1)
    manifest = []
    for batch in tqdm(loader, desc=f"inference, {args.num_steps} steps"):
        batch = batch.to(device)
        pred_ca, _ = integrate(model, batch, args.num_steps)
        pid, seq = batch.pair_id[0], sequence(batch.x)
        entry = {"pair_id": pid, "sequence_length": len(seq), "tracks": {}}
        for kind in tracks:
            ca = {"pred": pred_ca, "apo": batch.pos_0, "holo": batch.pos_T}[kind]
            ca_pdb = os.path.join(args.output_dir, "ca_only_pdbs", kind, f"{pid}.pdb")
            write_pdb(ca_pdb, ca.cpu().numpy(), seq)
            entry["tracks"][kind] = {"ca_only_pdb": ca_pdb,
                                     "allatom_pdb": os.path.join(args.output_dir, "allatom_pdbs", kind, f"{pid}.pdb")}
        manifest.append(entry)
    return manifest


def run_cg2all(job):
    """convert_cg2all on one Cα-only PDB; returns (pair_id, track, ok, message)."""
    pair_id, kind, in_pdb, out_pdb, cg2all_bin = job
    cmd = [cg2all_bin, "-p", in_pdb, "-o", out_pdb, "--cg", "CalphaBasedModel", "--device", "cpu"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as e:
        return pair_id, kind, False, str(e)
    ok = r.returncode == 0 and os.path.exists(out_pdb)
    return pair_id, kind, ok, "" if ok else (r.stderr[-500:] if r.stderr else r.stdout[-500:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint")
    ap.add_argument("--num_steps", type=int, default=10, help="Euler steps (the reported results use 10)")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--emb_dir", default=EMB_DIR)
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--output_dir", default="outputs/allatom")
    ap.add_argument("--tracks", default=",".join(TRACKS), help="comma-separated subset of pred,apo,holo")
    ap.add_argument("--cg2all_bin", default=os.environ.get("CONVERT_CG2ALL", "convert_cg2all"))
    ap.add_argument("--cg2all_workers", type=int, default=4, help="parallel cg2all processes")
    ap.add_argument("--skip_inference", action="store_true",
                    help="reuse the Cα-only PDBs in <output_dir>/manifest.json (cg2all on a CPU-only machine)")
    ap.add_argument("--skip_cg2all", action="store_true", help="only write the Cα-only PDBs")
    args = ap.parse_args()

    tracks = [t for t in args.tracks.split(",") if t]
    if set(tracks) - set(TRACKS):
        ap.error(f"--tracks must be a subset of {','.join(TRACKS)}")
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    if args.skip_inference:
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        if not args.checkpoint:
            ap.error("--checkpoint is required unless --skip_inference")
        manifest = write_ca_pdbs(args, tracks)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"wrote {len(manifest)} pairs x {len(tracks)} tracks of Cα-only PDBs, {manifest_path}")
    if args.skip_cg2all:
        return

    for kind in tracks:
        os.makedirs(os.path.join(args.output_dir, "allatom_pdbs", kind), exist_ok=True)
    jobs = [(e["pair_id"], kind, t["ca_only_pdb"], t["allatom_pdb"], args.cg2all_bin)
            for e in manifest for kind in tracks
            if (t := e["tracks"].get(kind)) and not os.path.exists(t["allatom_pdb"])]
    successes, fails = {kind: 0 for kind in tracks}, []
    with ThreadPoolExecutor(max_workers=args.cg2all_workers) as ex:
        for fut in tqdm(as_completed([ex.submit(run_cg2all, j) for j in jobs]), total=len(jobs), desc="cg2all"):
            pair_id, kind, ok, msg = fut.result()
            if ok:
                successes[kind] += 1
            else:
                fails.append({"pair_id": pair_id, "kind": kind, "msg": msg})
    print(f"cg2all: {successes} rebuilt, {len(fails)} failed")
    if fails:
        with open(os.path.join(args.output_dir, "cg2all_failures.json"), "w") as f:
            json.dump(fails, f, indent=2)


if __name__ == "__main__":
    main()
