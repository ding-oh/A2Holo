"""Sublayer update magnitude (R, rho_perp) and readout alignment (C) of the structure stream in each
double-stream block; --blocks all continues through the single-stream blocks, whose sublayer metrics
are NaN because a single block has no state between attention and MLP. Writes <tag>.tsv (per model
and block) and <tag>_perpair.tsv.

    python analysis/run.py --script analysis/update_alignment.py --t 0.5 \\
        --checkpoint seed1=checkpoints/a2holo_seed1 --max_pairs 150 --tag update_alignment_reference_set
    python analysis/run.py --script analysis/update_alignment.py --t 0.5 --blocks all \\
        --checkpoint seed1=checkpoints/a2holo_seed1 --max_pairs 150 --tag update_alignment_allblocks
"""
import os, sys, gc, argparse
import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, test_pairs, load_pair, MODELS, T_VAL, MAX_LEN, OUT_DIR as OUT, rho_perp, kabsch_rotation, cos_agg  # noqa: E402
from a2holo.model import attention  # noqa: E402

# Sublayer quantities a single block cannot define.
UNDEF_SINGLE = {"R_attn", "R_mlp", "rperp_attn", "rperp_mlp", "ndH_attn", "ndH_mlp",
                "C_attn", "C_mlp", "C_attn_perres"}


class UpdateCapturer:
    """Recompute each double block's structure-stream updates, gates included, from its forward kwargs:
    input h_in, attention update d_attn, state before the MLP h_pre_mlp, MLP update d_mlp, output h_out.
    With single=True also keep the structure half ([seq; struct], second half) of each single block's
    input and output."""

    def __init__(self, model, single=False):
        self.model = model
        self.single = single
        self.captures = {}
        self._hooks = []
        self._cond = None
        self.n_double = 0

    def register(self):
        for i, block in enumerate(self.model.double_blocks):
            self._hooks.append(
                block.register_forward_hook(self._make_hook(i), with_kwargs=True))
        self.n_double = len(self.model.double_blocks)
        if self.single:
            for j, block in enumerate(self.model.single_blocks):
                self._hooks.append(
                    block.register_forward_hook(self._single_hook(self.n_double + j)))

    def _single_hook(self, li):
        def hook(module, inputs, output):
            x_in = inputs[0]
            x_out = output[0] if isinstance(output, tuple) else output
            assert x_in.shape[1] % 2 == 0, f"odd token count {x_in.shape}"
            L = x_in.shape[1] // 2
            self.captures[li] = dict(h_in=x_in.detach()[:, L:, :],
                                     d_attn=None, h_pre_mlp=None, d_mlp=None,
                                     h_out=x_out.detach()[:, L:, :],
                                     cond=self._cond)
        return hook

    def _make_hook(self, li):
        def hook(module, inputs, kwargs, output):
            struct, seq, cond, pe = kwargs["struct"], kwargs["seq"], kwargs["cond"], kwargs["pe"]
            H = module.num_heads
            struct_mod1, struct_mod2 = module.struct_mod(cond)
            seq_mod1, _ = module.seq_mod(cond)

            sm = (1 + struct_mod1.scale) * module.struct_norm1(struct) + struct_mod1.shift
            sq, sk, sv = rearrange(module.struct_attn.qkv(sm),
                                   "B L (K H D) -> K B H L D", K=3, H=H)
            sq, sk = module.struct_attn.norm(sq, sk, sv)
            qm = (1 + seq_mod1.scale) * module.seq_norm1(seq) + seq_mod1.shift
            qq, qk, qv = rearrange(module.seq_attn.qkv(qm),
                                   "B L (K H D) -> K B H L D", K=3, H=H)
            qq, qk = module.seq_attn.norm(qq, qk, qv)

            att = attention(
                torch.cat((qq, sq), dim=2), torch.cat((qk, sk), dim=2),
                torch.cat((qv, sv), dim=2), pe=pe)
            struct_attn_out = att[:, seq.shape[1]:]

            d_attn = struct_mod1.gate * module.struct_attn.proj(struct_attn_out)
            h_pre_mlp = struct + d_attn
            d_mlp = struct_mod2.gate * module.struct_mlp(
                (1 + struct_mod2.scale) * module.struct_norm2(h_pre_mlp) + struct_mod2.shift)

            self._cond = cond.detach()
            self.captures[li] = dict(h_in=struct.detach(), d_attn=d_attn.detach(),
                                     h_pre_mlp=h_pre_mlp.detach(), d_mlp=d_mlp.detach(),
                                     h_out=(h_pre_mlp + d_mlp).detach(),
                                     cond=cond.detach())
        return hook

    def check(self):
        """Raise unless the last forward captured every block the capturer hooked."""
        n = len(self.model.double_blocks) + (len(self.model.single_blocks) if self.single else 0)
        if len(self.captures) != n:
            raise RuntimeError(f"captured {len(self.captures)} of {n} blocks; "
                               "the update hooks did not run")

    def clear(self):
        self.captures = {}

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", choices=("double", "all"), default="double",
                    help="double-stream blocks only, or every block (double then single)")
    ap.add_argument("--max_pairs", type=int, default=150)
    ap.add_argument("--tag", default="update_alignment")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.join(OUT, args.tag)), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    every = args.blocks == "all"

    pairs = test_pairs()
    print(f"[env] device={device}  candidate pairs={len(pairs)}  blocks={args.blocks}", flush=True)

    # R = ||dH|| / ||H||; rho_perp as above; C = cos_agg(dv @ Rot, x_holo - x_apo) with dv the change a
    # sublayer makes to the readout P(H) = final_layer(H, cond)[:, :3] (a difference of readouts, since
    # final_layer is not linear); exit_* read P(h_out) directly. Rot superposes the model's final
    # prediction onto holo, one per pair; C_final (under Rot) and C_final_noR (without) are the controls.
    # Over every block, exit_normratio = ||P(h_out) @ Rot|| / ||dx|| and dv_normratio = ||dv_block @ Rot|| / ||dx||.
    METRICS = (["R_attn", "R_mlp", "R_block", "rperp_attn", "rperp_mlp", "rperp_block",
                "nH_in", "ndH_attn", "ndH_mlp", "C_attn", "C_mlp", "C_block",
                "exit_cos", "exit_relerr"]
               + (["exit_normratio", "dv_normratio"] if every else [])
               + ["C_final", "C_final_noR", "C_attn_perres"])
    rows = []
    perpair = []

    for label, ckdir in MODELS:
        path = os.path.join(ckdir, "best_model.pt")
        if not os.path.exists(path):
            print(f"[skip] {label}: no checkpoint", flush=True)
            continue
        model, n_layers = load_model(path, device)
        torch.cuda.empty_cache()
        cap = UpdateCapturer(model, single=every)
        cap.register()
        print(f"[model] {label:<18s} depth_double={n_layers}", flush=True)

        n_single = len(model.single_blocks)
        n_blocks = n_layers + n_single if every else n_layers
        acc = {m: [[] for _ in range(n_blocks)] for m in METRICS}
        used = 0
        for pid in pairs:
            if used >= args.max_pairs:
                break
            p = load_pair(pid, device)
            if p["n"] > MAX_LEN:
                continue
            N, ca0, caT = p["n"], p["ca0"], p["caT"]

            pos_ca = (1 - T_VAL) * ca0 + T_VAL * caT
            pos_cb = (1 - T_VAL) * p["cb0"] + T_VAL * p["cbT"]
            t = torch.full((1,), T_VAL, device=device)
            cap.clear()
            with torch.no_grad():
                v_ca, _ = model(p["x"], pos_ca, pos_cb, t, p["bidx"], p["seq_emb"], p["global_token"])
            cap.check()

            dx = (caT - ca0).double()
            with torch.no_grad():
                R = kabsch_rotation((ca0 + v_ca).double(), caT.double())
                c_final = cos_agg(v_ca.double() @ R.double(), dx)
                c_final_noR = cos_agg(v_ca.double(), dx)

            with torch.no_grad():
                for li in range(n_blocks):
                    c = cap.captures.get(li)
                    if c is None:
                        continue
                    h_in, d_a, h_pm, d_m, h_out, cond = (
                        c["h_in"], c["d_attn"], c["h_pre_mlp"], c["d_mlp"],
                        c["h_out"], c["cond"])
                    is_single = d_a is None
                    d_b = h_out - h_in
                    a = acc
                    a["nH_in"][li].append(h_in.norm().item())
                    a["R_block"][li].append((d_b.norm() / h_in.norm()).item())
                    a["rperp_block"][li].append(rho_perp(d_b, h_in))
                    if not is_single:
                        a["ndH_attn"][li].append(d_a.norm().item())
                        a["ndH_mlp"][li].append(d_m.norm().item())
                        a["R_attn"][li].append((d_a.norm() / h_in.norm()).item())
                        a["R_mlp"][li].append((d_m.norm() / h_pm.norm()).item())
                        a["rperp_attn"][li].append(rho_perp(d_a, h_in))
                        a["rperp_mlp"][li].append(rho_perp(d_m, h_pm))

                    P = lambda H: model.final_layer(H, cond)[0, :, :3].double()
                    p_in, p_out = P(h_in), P(h_out)
                    a["C_block"][li].append(cos_agg((p_out - p_in) @ R.double(), dx))
                    if not is_single:
                        p_pm = P(h_pm)
                        a["C_attn"][li].append(cos_agg((p_pm - p_in) @ R.double(), dx))
                        a["C_mlp"][li].append(cos_agg((p_out - p_pm) @ R.double(), dx))
                        dva = (p_pm - p_in) @ R.double()
                        pr = (dva * dx).sum(-1) / (
                            dva.norm(dim=-1).clamp_min(1e-12) * dx.norm(dim=-1).clamp_min(1e-12))
                        a["C_attn_perres"][li].append(pr.mean().item())
                    a["C_final"][li].append(c_final)
                    a["C_final_noR"][li].append(c_final_noR)
                    ve = p_out @ R.double()
                    if every:
                        a["exit_normratio"][li].append(
                            (ve.norm() / dx.norm().clamp_min(1e-12)).item())
                        a["dv_normratio"][li].append(
                            (((p_out - p_in) @ R.double()).norm()
                             / dx.norm().clamp_min(1e-12)).item())
                    a["exit_cos"][li].append(cos_agg(ve, dx))
                    a["exit_relerr"][li].append(
                        ((ve - dx).norm() / dx.norm().clamp_min(1e-12)).item())
                    rec = dict(model=label, pair=pid, N=N, depth=n_layers)
                    if every:
                        rec["n_single"] = n_single
                        rec["block_type"] = "single" if is_single else "double"
                    rec["layer"] = li
                    rec.update({m: (float("nan") if is_single and m in UNDEF_SINGLE
                                    else acc[m][li][-1]) for m in METRICS})
                    perpair.append(rec)
            used += 1
            cap.clear()
            del p
            gc.collect()

        for li in range(n_blocks):
            r = dict(model=label, depth=n_layers, n_single=n_single, layer=li,
                     block_type="double" if li < n_layers else "single",
                     layer_frac=li / max(n_blocks - 1, 1), n=used)
            for m in METRICS:
                r[m] = float(np.mean(acc[m][li])) if acc[m][li] else np.nan
            rows.append(r)

        cur = rows[-n_blocks:]
        for m in ("R_block", "rperp_block", "C_attn", "C_attn_perres", "C_final", "exit_relerr"):
            v = [cur[i][m] for i in range(n_blocks)]
            print(f"[{label}] {m:<12s} depth-mean={np.nanmean(v):+.4f}  "
                  f"argmax=L{int(np.nanargmax(v))} argmin=L{int(np.nanargmin(v))} "
                  f"(indices are NOT a claim)", flush=True)
        cap.remove()
        cap.clear()
        del model, cap
        torch.cuda.empty_cache()
        gc.collect()

    keys = (["model", "depth"] + (["n_single", "block_type"] if every else [])
            + ["layer", "layer_frac", "n"] + METRICS)
    tsv = os.path.join(OUT, f"{args.tag}.tsv")
    with open(tsv, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")
    print(f"[done] wrote {tsv}", flush=True)
    pp = os.path.join(OUT, f"{args.tag}_perpair.tsv")
    pkeys = list(perpair[0]) if perpair else []
    with open(pp, "w") as f:
        f.write("\t".join(pkeys) + "\n")
        for r in perpair:
            f.write("\t".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                              for k in pkeys) + "\n")
    print(f"[done] wrote {pp}  ({len(perpair)} pair x block rows)", flush=True)


if __name__ == "__main__":
    main()
