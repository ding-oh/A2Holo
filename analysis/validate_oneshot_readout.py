"""
Gate before the analyses: the analysis forward pass must reproduce the evaluated model.

At t = 0 the analysis prediction ca0 + v(ca0, 0) is exactly one left-endpoint Euler step,
which is what a one-step evaluation scores.  The per-pair CA RMSD from this forward pass
must match ca_rmsd in that evaluation's JSON to within --tol; the script exits otherwise,
so analyses chained after it run only on a checkpoint it verified.  This ties the analysed
weights to the benchmark number.  It says nothing about t > 0, where no evaluation exists
to compare.

Writes readout_t0.tsv (per pair: RMSD from the t = 0 analysis pass and from the one-step
evaluation) and checkpoint_pin.json to --out_dir.  Run from the repository root on a GPU:

    python scripts/evaluate.py --checkpoint checkpoints/a2holo_seed1/best_model.pt \\
        --num_steps 1 --output outputs/analysis/gate_seed1/nfe1_results.json
    python analysis/validate_oneshot_readout.py --ckdir checkpoints/a2holo_seed1 \\
        --eval_json outputs/analysis/gate_seed1/nfe1_results.json \\
        --out_dir outputs/analysis/gate_seed1
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, test_pairs, load_pair, MAX_LEN, kabsch_rmsd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckdir", required=True, help="directory holding best_model.pt")
    ap.add_argument("--eval_json", required=True,
                    help="one-step test evaluation JSON (scripts/evaluate.py --num_steps 1 --output FILE)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="largest per-pair |RMSD difference| in A accepted")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    pin_path = os.path.join(out_dir, "checkpoint_pin.json")
    ck_path = os.path.join(args.ckdir, "best_model.pt")

    if not torch.cuda.is_available():
        sys.exit("no GPU visible -- this script needs a GPU")
    device = "cuda"

    ev_mtime = os.stat(args.eval_json).st_mtime
    if os.stat(ck_path).st_mtime > ev_mtime:
        sys.exit(f"checkpoint {ck_path} is newer than the eval JSON; the eval "
                 "scored an earlier file")
    ck = torch.load(ck_path, map_location="cpu", mmap=True, weights_only=True)
    meta = dict(epoch=ck.get("epoch"), config=ck["config"], train_args=ck.get("train_args"))
    del ck
    print(f"[identity] {json.dumps(meta)}", flush=True)
    model, n_double = load_model(ck_path, device)

    ev = json.load(open(args.eval_json))
    if int(ev.get("num_steps", -1)) != 1 or ev.get("split") != "test":
        sys.exit(f"eval JSON is not one-step / test: "
                 f"{ {k: ev.get(k) for k in ('num_steps', 'split')} }")
    ref = dict(zip(ev["pair_ids"], ev["ca_rmsd"]))

    rows = []
    for pid in test_pairs():
        pair = load_pair(pid, device)
        N = pair["n"]
        if N > MAX_LEN:
            continue
        ca0 = pair["ca0"]
        t = torch.zeros((1,), device=device)
        with torch.no_grad():
            v_ca, _ = model(pair["x"], ca0, pair["cb0"], t, pair["bidx"],
                            pair["seq_emb"], pair["global_token"])
        ours = kabsch_rmsd((ca0 + v_ca).double(), pair["caT"].double())
        if pid not in ref:
            sys.exit(f"{pid} is in the analysis subset but not in the eval JSON")
        rows.append((pid, N, ours, float(ref[pid])))
        del pair

    ours = np.array([r[2] for r in rows])
    theirs = np.array([r[3] for r in rows])
    diff = np.abs(ours - theirs)
    worst = sorted(rows, key=lambda r: -abs(r[2] - r[3]))[:5]
    summary = dict(
        n_pairs=len(rows), max_len=MAX_LEN,
        median_rmsd_ours=float(np.median(ours)), median_rmsd_eval_same_pairs=float(np.median(theirs)),
        mean_rmsd_ours=float(ours.mean()), mean_rmsd_eval_same_pairs=float(theirs.mean()),
        eval_median_all=float(ev["ca"]["median"]), eval_n_all=int(ev["ca"]["n"]),
        abs_diff_median=float(np.median(diff)), abs_diff_max=float(diff.max()),
        worst=[dict(pair=w[0], N=w[1], ours=w[2], eval=w[3]) for w in worst])
    print(f"[readout] {json.dumps(summary)}", flush=True)

    with open(os.path.join(out_dir, "readout_t0.tsv"), "w") as fh:
        fh.write("pair\tN\trmsd_analysis_t0\trmsd_eval_nfe1\n")
        for pid, N, o, e in rows:
            fh.write(f"{pid}\t{N}\t{o:.6f}\t{e:.6f}\n")
    with open(pin_path, "w") as fh:
        json.dump(dict(meta=meta, eval_json=args.eval_json,
                       eval_json_mtime=time.strftime("%Y-%m-%d %H:%M:%S",
                                                     time.localtime(ev_mtime)),
                       readout=summary, n_double=n_double), fh, indent=1)

    if diff.max() > args.tol:
        sys.exit(f"READOUT MISMATCH: max |diff| {diff.max():.4f} A > tol {args.tol}; "
                 "the analysis forward pass is not the benchmarked model")
    print("[gate] PASS", flush=True)


if __name__ == "__main__":
    main()
