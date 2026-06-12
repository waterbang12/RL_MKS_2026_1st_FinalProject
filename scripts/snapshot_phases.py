"""
Snapshot the reference episode to identify phase boundaries.

Usage:
    python scripts/snapshot_phases.py
    python scripts/snapshot_phases.py --seq data/HOCAP/sequence1/sequence1.pt
    python scripts/snapshot_phases.py --seq data/HOCAP/sequence2/sequence2.pt --table_z 0.4

Outputs:
    - phase_snapshot.png  (saved next to the .pt file)
    - printed table of per-frame key metrics
"""

import argparse
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

MANO_FINGERTIPS = [4, 8, 12, 16, 20]
FINGER_NAMES    = ["thumb", "index", "middle", "ring", "little"]

def load_seq(path: str, table_z: float = 0.4):
    data = torch.load(path, map_location="cpu")

    obj_bottom_offset = data["obj_bottom_offset"].item()
    obj_reset_z = table_z + obj_bottom_offset
    obj_trans   = data["obj_trans"]           # (T, 3)
    to_center   = -obj_trans[0:1] + torch.tensor([[0.0, 0.0, obj_reset_z]])

    obj_pos  = obj_trans + to_center          # (T, 3)  world-space
    obj_vel  = data["obj_vel"]                # (T, 3)
    mano     = data["mano_kpts"]              # (T, 21, 3)
    mano_pos = mano + to_center.unsqueeze(1)  # (T, 21, 3)
    tips     = mano_pos[:, MANO_FINGERTIPS]   # (T, 5, 3)
    wrist    = mano_pos[:, 0]                 # (T, 3)

    return obj_pos, obj_vel, tips, wrist

def compute_metrics(obj_pos, obj_vel, tips, wrist, table_z=0.4):
    T = obj_pos.shape[0]
    obj_z      = obj_pos[:, 2]                                  # (T,)
    lift       = (obj_z - table_z).clamp(min=0.0)              # (T,)
    vel_z      = obj_vel[:, 2]                                  # (T,)
    tip_dists  = torch.norm(tips - obj_pos.unsqueeze(1), dim=-1)  # (T, 5)
    min_dist   = tip_dists.min(dim=-1).values                   # (T,)
    wrist_z    = wrist[:, 2]                                    # (T,)
    frames     = torch.arange(T)
    return frames, obj_z, lift, vel_z, tip_dists, min_dist, wrist_z

def find_phase_boundaries(lift, vel_z, tip_dists, obj_z, table_z=0.4):
    T = lift.shape[0]
    LIFT_THRESH   = 0.005   # 5 mm off table
    VEL_THRESH    = 0.02    # 2 cm/s upward
    CONTACT_THRESH = 0.06   # within 6 cm of obj center (capsule radius ~4 cm)

    # lift-off: first frame obj is meaningfully above table
    lift_frame = next((t for t in range(T) if lift[t] > LIFT_THRESH), T - 1)

    # contact: all 5 tips within CONTACT_THRESH simultaneously, sustained for 3 frames
    all_contact = (tip_dists < CONTACT_THRESH).all(dim=-1)  # (T,)
    contact_frame = T - 1
    for t in range(T - 2):
        if all_contact[t] and all_contact[t+1] and all_contact[t+2]:
            contact_frame = t
            break

    return contact_frame, lift_frame

def print_table(frames, obj_z, lift, vel_z, tip_dists, min_dist, wrist_z,
                contact_frame, lift_frame, table_z, every=5):
    print(f"\n{'Frame':>5}  {'obj_z':>7}  {'lift_mm':>7}  {'vel_z':>7}  "
          f"{'min_dist':>8}  {'wrist_z':>7}  {'phase':>8}")
    print("-" * 65)
    T = len(frames)
    for t in range(0, T, every):
        phase = "approach"
        if t >= contact_frame:
            phase = "grasp"
        if t >= lift_frame:
            phase = "lift"
        marker = " <--" if t in (contact_frame, lift_frame) else ""
        print(f"{t:>5}  {obj_z[t]:>7.4f}  {lift[t]*1000:>7.1f}  "
              f"{vel_z[t]:>7.4f}  {min_dist[t]:>8.4f}  {wrist_z[t]:>7.4f}  "
              f"{phase:>8}{marker}")
    print(f"\nDetected boundaries:  contact={contact_frame}  lift-off={lift_frame}")

def plot(frames, obj_z, lift, vel_z, tip_dists, min_dist, wrist_z,
         contact_frame, lift_frame, table_z, out_path):
    T = len(frames)
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    def vlines(ax):
        ax.axvline(contact_frame, color="green",  lw=1.5, ls="--", label=f"contact t={contact_frame}")
        ax.axvline(lift_frame,    color="red",    lw=1.5, ls="--", label=f"lift-off t={lift_frame}")

    # --- obj z height ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(frames, obj_z.numpy(), color="steelblue")
    ax.axhline(table_z, color="gray", lw=1, ls=":")
    vlines(ax)
    ax.set_title("Object height (z)")
    ax.set_xlabel("frame"); ax.set_ylabel("m")
    ax.legend(fontsize=7)

    # --- lift above table ---
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(frames, lift.numpy() * 1000, color="orange")
    vlines(ax)
    ax.set_title("Lift above table (mm)")
    ax.set_xlabel("frame"); ax.set_ylabel("mm")
    ax.legend(fontsize=7)

    # --- object vertical velocity ---
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(frames, vel_z.numpy(), color="purple")
    ax.axhline(0, color="gray", lw=1)
    vlines(ax)
    ax.set_title("Object vel_z (m/s)")
    ax.set_xlabel("frame"); ax.set_ylabel("m/s")
    ax.legend(fontsize=7)

    # --- per-fingertip distance to object center ---
    ax = fig.add_subplot(gs[1, 1])
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    for i, (name, col) in enumerate(zip(FINGER_NAMES, colors)):
        ax.plot(frames, tip_dists[:, i].numpy(), color=col, label=name, lw=1)
    ax.axhline(0.04, color="gray", lw=1, ls=":", label="capsule r=4cm")
    vlines(ax)
    ax.set_title("Fingertip distance to obj center (m)")
    ax.set_xlabel("frame"); ax.set_ylabel("m")
    ax.legend(fontsize=6, ncol=2)

    # --- min distance across all tips ---
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(frames, min_dist.numpy(), color="black")
    ax.axhline(0.04, color="gray", lw=1, ls=":", label="capsule r=4cm")
    vlines(ax)
    ax.set_title("Min fingertip distance to obj (m)")
    ax.set_xlabel("frame"); ax.set_ylabel("m")
    ax.legend(fontsize=7)

    # --- wrist height ---
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(frames, wrist_z.numpy(), color="teal")
    ax.plot(frames, obj_z.numpy(), color="steelblue", ls="--", lw=1, label="obj_z")
    vlines(ax)
    ax.set_title("Wrist height (z)")
    ax.set_xlabel("frame"); ax.set_ylabel("m")
    ax.legend(fontsize=7)

    # shade phases
    for ax in fig.get_axes():
        ax.axvspan(0,            contact_frame, alpha=0.06, color="blue",  label="approach")
        ax.axvspan(contact_frame, lift_frame,   alpha=0.06, color="green", label="grasp")
        ax.axvspan(lift_frame,   T,             alpha=0.06, color="red",   label="lift")

    fig.suptitle(f"Reference episode phase snapshot  |  contact={contact_frame}  lift={lift_frame}",
                 fontsize=12, fontweight="bold")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",     default="data/HOCAP/sequence1/sequence1.pt")
    parser.add_argument("--table_z", type=float, default=0.4)
    parser.add_argument("--every",   type=int,   default=5,
                        help="print a row every N frames")
    args = parser.parse_args()

    seq_path = Path(args.seq)
    obj_pos, obj_vel, tips, wrist = load_seq(str(seq_path), args.table_z)
    frames, obj_z, lift, vel_z, tip_dists, min_dist, wrist_z = compute_metrics(
        obj_pos, obj_vel, tips, wrist, args.table_z
    )
    contact_frame, lift_frame = find_phase_boundaries(lift, vel_z, tip_dists, obj_z, args.table_z)

    print_table(frames, obj_z, lift, vel_z, tip_dists, min_dist, wrist_z,
                contact_frame, lift_frame, args.table_z, args.every)

    out_path = seq_path.parent / "phase_snapshot.png"
    plot(frames, obj_z, lift, vel_z, tip_dists, min_dist, wrist_z,
         contact_frame, lift_frame, args.table_z, out_path)

if __name__ == "__main__":
    main()
