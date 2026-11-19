#!/usr/bin/env python3
"""The animated attention figure: one pair carried from apo to the model's prediction.

One frame per Euler step of the model's own trajectory. Four panels share one protein, one camera
and one colour scale; colour is a per-residue attention quantity, size is the fixed apo-to-holo
displacement, so a bright large dot is a residue that is both attended and moving.

The four quantities are read off the joint attention map A of shape 2N x 2N, head-averaged and then
averaged over every double-stream block, with tokens ordered [sequence; structure]:

    received      A[N:, N:].mean(axis=0)    how much a structure residue is read
    emitted       A[N:, N:].mean(axis=1)    how much it reads
    seq -> str    A[:N, N:].mean(axis=0)    what the sequence stream pays to structure residue i
    seq -> seq    A[:N, :N].mean(axis=0)    what it pays to sequence token i

Rows of A sum to 1, so each quantity is a share of attention; the run stops if they do not.

Run:
    python analysis/attention_gif.py --checkpoint checkpoints/a2holo_seed1/best_model.pt
    python analysis/attention_gif.py --checkpoint <ckpt> --pair_id 2IZD_B_5B5F_C --nfe 10
    python analysis/attention_gif.py --npz <out>.npz --norm per_panel --out redrawn.gif

Integrating in more steps than the deployed ten only spaces the same path more finely, and makes a
smoother animation. Writes --out and nothing else.
"""
import argparse
import os
import sys
from io import BytesIO

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AttentionCapturer, load_model, load_pair  # noqa: E402

BG = "#080c12"
TARGET_C = "#4da6ff"
APO_C = "#a0aab4"
PRED_C = "#ff6b35"
TEXT_C = "#f0f6fc"
CMAP = plt.cm.inferno
MONO = "DejaVu Sans Mono"

PANELS = [
    ("recv", "Structure stream, received"),
    ("emit", "Structure stream, emitted"),
    ("seq2str", "Sequence → structure"),
    ("seq2seq", "Sequence → sequence"),
]

# Figure-fraction geometry of the four panels, their titles, the frame title, the two footer lines
# and the colour bar.
AX_RECT = [(0.00, 0.500, 0.50, 0.372), (0.50, 0.500, 0.50, 0.372),
           (0.00, 0.093, 0.50, 0.372), (0.50, 0.093, 0.50, 0.372)]
AX_TITLE = [(0.25, 0.87759), (0.75, 0.87759), (0.25, 0.47069), (0.75, 0.47069)]
TITLE_Y = 0.95776
FOOTER_Y = (0.08650, 0.06650)
FOOTER = ("emitted = what a residue reads      received = how much it is read",
          "dot size = apo-to-holo displacement")
CBAR_RECT = (0.30, 0.01640, 0.40, 0.01900)


def kabsch_np(P, Q):
    """P superposed onto Q, with the rotation and the two centroids."""
    cP, cQ = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cP).T @ (Q - cQ))
    d = np.linalg.det(Vt.T @ U.T)
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return (P - cP) @ R + cQ, R, cP, cQ


def fixed_basis(holo_c, rot_deg):
    """First two principal axes of holo, sign-fixed and right-handed, rotated in plane by rot_deg.

    Eigenvector signs are arbitrary, so each is pinned to its largest component; without that the
    trace can be drawn as its own mirror image.
    """
    _, V = np.linalg.eigh(holo_c.T @ holo_c / len(holo_c))
    e1, e2 = V[:, -1].copy(), V[:, -2].copy()
    for e in (e1, e2):
        if e[np.argmax(np.abs(e))] < 0:
            e *= -1
    e2 = e2 - (e2 @ e1) * e1
    e2 /= np.linalg.norm(e2)
    a = np.deg2rad(rot_deg)
    return np.cos(a) * e1 - np.sin(a) * e2, np.sin(a) * e1 + np.cos(a) * e2


def block_mean_map(cap):
    """Attention of every double-stream block, averaged over blocks, with its token count."""
    cap.check()
    idx = sorted(cap.captures)
    L = cap.captures[idx[0]]["L"]
    A = np.mean([cap.captures[i]["attn"] for i in idx], axis=0).astype(np.float64)
    if A.shape[0] != 2 * L:
        raise RuntimeError(f"attention map {A.shape} is not 2N x 2N with N={L}")
    rs = A.sum(axis=1)
    if not (0.98 < rs.min() and rs.max() < 1.02):
        raise RuntimeError(f"attention rows are not normalised: {rs.min():.4f}-{rs.max():.4f}")
    return A, L


def four_quantities(A, L):
    cc = A[L:, L:]
    return dict(recv=cc.mean(axis=0), emit=cc.mean(axis=1),
                seq2str=A[:L, L:].mean(axis=0), seq2seq=A[:L, :L].mean(axis=0))


def capture_trajectory(model, pair, nfe, device):
    """Integrate from apo in `nfe` Euler steps, capturing the four quantities at every step."""
    cap = AttentionCapturer(model)
    cap.register()
    pos_ca, pos_cb = pair["ca0"].clone(), pair["cb0"].clone()
    dt = 1.0 / nfe
    t_traj, ca_traj, Q = [], [], {k: [] for k, _ in PANELS}
    for step in range(nfe + 1):
        # The model's own t input is clamped just short of 1, the range it was trained on.
        tv = 1.0 if step == nfe else step * dt
        t = torch.full((1,), min(tv, 0.999), device=device)
        cap.clear()
        with torch.no_grad():
            v_ca, v_cb = model(pair["x"], pos_ca, pos_cb, t, pair["bidx"],
                               pair["seq_emb"], pair["global_token"])
        A, L = block_mean_map(cap)
        for k, v in four_quantities(A, L).items():
            Q[k].append(v)
        t_traj.append(tv)
        ca_traj.append(pos_ca.detach().cpu().numpy().astype(np.float64))
        if step < nfe:
            pos_ca = pos_ca + dt * v_ca
            pos_cb = pos_cb + dt * v_cb
        print(f"[step] t={tv:.3f}", flush=True)
    cap.remove()
    return np.array(t_traj), np.stack(ca_traj), {k: np.stack(v) for k, v in Q.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/a2holo_seed1/best_model.pt")
    ap.add_argument("--pair_id", default="4AKE_A_4X8L_A")
    ap.add_argument("--nfe", type=int, default=60, help="Euler steps, one frame each")
    ap.add_argument("--out", default="outputs/analysis/attention_displacement.gif")
    ap.add_argument("--npz", default=None,
                    help="reuse a previous run's captured arrays and skip the model entirely; "
                         "every run writes <out>.npz, so the drawing can be changed for free")
    ap.add_argument("--rot", type=float, default=0.0, help="in-plane rotation of the fixed basis")
    ap.add_argument("--zoom", type=float, default=1.0, help=">1 draws the protein larger")
    ap.add_argument("--margin", type=float, default=1.22)
    ap.add_argument("--smin", type=float, default=13.0, help="marker area of the least-moved Ca")
    ap.add_argument("--sspan", type=float, default=132.0, help="area added at the largest")
    ap.add_argument("--hold", type=int, default=8, help="frames held on the last step")
    ap.add_argument("--duration", type=int, default=60, help="ms per frame")
    ap.add_argument("--render_px", type=int, default=1560)
    ap.add_argument("--out_px", type=int, default=1160)
    ap.add_argument("--norm", default="shared", choices=("shared", "per_panel"),
                    help="shared: one colour range over all four panels, the only mode under which "
                         "they compare. per_panel: each panel stretched to its own min-max")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    if a.npz:
        z = np.load(a.npz)
        if str(z["pair"]) != a.pair_id:
            raise SystemExit(f"{a.npz} holds pair {z['pair']}, not {a.pair_id}")
        t_traj, traj = z["t_traj"], z["ca_traj"]
        apo, holo = z["ca_apo"], z["ca_holo"]
        Q = {k: z[k] for k, _ in PANELS}
        N = len(apo)
        print(f"[env] reusing {a.npz}  pair={a.pair_id}  N={N}  {len(t_traj)} steps", flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, depth_double = load_model(a.checkpoint, device)
        pair = load_pair(a.pair_id, device)
        N = pair["n"]
        print(f"[env] device={device}  pair={a.pair_id}  N={N}  "
              f"double-stream depth={depth_double}", flush=True)
        t_traj, traj, Q = capture_trajectory(model, pair, a.nfe, device)
        apo = pair["ca0"].cpu().numpy().astype(np.float64)
        holo = pair["caT"].cpu().numpy().astype(np.float64)
        np.savez_compressed(a.out + ".npz", pair=a.pair_id, t_traj=t_traj, ca_traj=traj,
                            ca_apo=apo, ca_holo=holo, **Q)
        print(f"[env] captured arrays written to {a.out}.npz", flush=True)
    disp = np.linalg.norm(holo - apo, axis=-1)

    # Every frame is superposed on holo and projected on one basis with one set of limits, shared by
    # all four panels and all frames. Without both the protein tumbles and the eye reads the tumble
    # as the motion.
    traj_a = np.stack([kabsch_np(f, holo)[0] for f in traj])
    apo_a = kabsch_np(apo, holo)[0]
    c = holo.mean(0)
    holo_c, apo_c, traj_c = holo - c, apo_a - c, traj_a - c
    rmsd_traj = np.sqrt(((traj_a - holo[None]) ** 2).sum(-1).mean(-1))
    e1, e2 = fixed_basis(holo_c, a.rot)
    project = lambda X: np.stack([X @ e1, X @ e2], axis=-1)  # noqa: E731
    holo2, apo2, traj2 = project(holo_c), project(apo_c), project(traj_c)

    pts = np.concatenate([traj2.reshape(-1, 2), holo2, apo2])
    ratio = AX_RECT[0][2] / AX_RECT[0][3]
    lo, hi = pts.min(0), pts.max(0)
    # The bounding-box centre, not the centroid: the open apo lid sits on one side and would
    # otherwise ride high in the frame.
    ctr, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    hh = max(half[1], half[0] / ratio) * a.margin / a.zoom
    hw = hh * ratio

    # One colour range over the four panels. Per-panel min-max gives every panel the full colourbar
    # however narrow its range, so the weakest quantity is drawn as vividly as the strongest.
    norm = {}
    if a.norm == "shared":
        glo = min(float(Q[k].min()) for k, _ in PANELS)
        ghi = max(float(Q[k].max()) for k, _ in PANELS)
    for key, _ in PANELS:
        klo, khi = float(Q[key].min()), float(Q[key].max())
        if a.norm == "shared":
            norm[key] = (Q[key] - glo) / max(ghi - glo, 1e-30)
            print(f"  {key:<8s} min {klo:.5f}  max {khi:.5f}  (shared scale {glo:.5f}-{ghi:.5f})")
        else:
            norm[key] = (Q[key] - klo) / max(khi - klo, 1e-30)
            print(f"  {key:<8s} min {klo:.5f}  max {khi:.5f}  (stretched to the full colourbar)")
    sizes = a.smin + (disp / disp.max()) ** 0.7 * a.sspan

    inch = a.render_px / 120.0
    fig = plt.figure(figsize=(inch, inch), facecolor=BG, dpi=120)
    title = fig.text(0.5, TITLE_Y, "", color=TEXT_C, fontsize=22, fontweight="bold",
                     fontfamily=MONO, ha="center", va="baseline")
    artists = []
    for i, (key, label) in enumerate(PANELS):
        ax = fig.add_axes(AX_RECT[i])
        ax.set_facecolor(BG)
        ax.set_axis_off()
        ax.set_xlim(ctr[0] - hw, ctr[0] + hw)
        ax.set_ylim(ctr[1] - hh, ctr[1] + hh)
        ax.plot(holo2[:, 0], holo2[:, 1], color=TARGET_C, lw=0.9, alpha=0.22, zorder=1)
        ax.plot(apo2[:, 0], apo2[:, 1], color=APO_C, lw=0.7, alpha=0.30,
                linestyle="--", dashes=(4, 3), zorder=2)
        chain, = ax.plot([], [], color=PRED_C, lw=1.0, alpha=0.50,
                         solid_capstyle="round", zorder=4)
        sc = ax.scatter(np.zeros(N), np.zeros(N), s=sizes, c=np.zeros(N), cmap=CMAP,
                        vmin=0.0, vmax=1.0, edgecolors=(1, 1, 1, 0.35), linewidths=0.5, zorder=6)
        fig.text(AX_TITLE[i][0], AX_TITLE[i][1], label, color=TEXT_C, fontsize=20,
                 fontweight="bold", fontfamily=MONO, ha="center", va="baseline")
        artists.append((key, chain, sc))

    for y, line in zip(FOOTER_Y, FOOTER):
        fig.text(0.5, y, line, color=TEXT_C, fontsize=14, fontweight="bold",
                 fontfamily=MONO, ha="center", va="baseline")
    cax = fig.add_axes(CBAR_RECT)
    cax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto", cmap=CMAP)
    cax.set_xticks([])
    cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)
    cax.text(-0.017, 0.5, "min", color=TEXT_C, fontsize=22, fontweight="bold", fontfamily=MONO,
             transform=cax.transAxes, ha="right", va="center")
    cax.text(1.015, 0.5, "max", color=TEXT_C, fontsize=22, fontweight="bold", fontfamily=MONO,
             transform=cax.transAxes, ha="left", va="center")

    frames = []
    for fi in range(len(t_traj) + a.hold):
        f = min(fi, len(t_traj) - 1)
        xy = traj2[f]
        for key, chain, sc in artists:
            chain.set_data(xy[:, 0], xy[:, 1])
            sc.set_offsets(xy)
            sc.set_array(norm[key][f])
        # The pair is named in the caption; what changes frame to frame is the flow time and how far
        # the structure still is from holo.
        title.set_text(f"t = {t_traj[f]:.2f}    Cα RMSD {rmsd_traj[f]:.2f} Å")
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=BG, edgecolor="none", dpi=fig.dpi)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
        if a.out_px != a.render_px:
            im = im.resize((a.out_px, a.out_px), Image.LANCZOS)
        frames.append(im.copy())
        buf.close()
    plt.close(fig)

    # One adaptive palette for the whole GIF, from a middle frame: quantising each frame on its own
    # makes the colourbar gradient, which never changes, shimmer.
    pal = frames[len(frames) // 2].convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(a.out, save_all=True, append_images=q[1:], duration=a.duration, loop=0,
              optimize=False, disposal=2)
    print(f"Cα RMSD to holo: {rmsd_traj[0]:.2f} -> {rmsd_traj[-1]:.2f} A")
    # Pillow merges runs of identical frames and adds their delays, so the held frames become one
    # frame with a long delay; the GIF runs for the same time.
    print(f"wrote {a.out} ({len(q)} rendered, {Image.open(a.out).n_frames} in the file, "
          f"{a.out_px}x{a.out_px}, {os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
