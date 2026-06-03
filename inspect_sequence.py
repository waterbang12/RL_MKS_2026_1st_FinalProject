import torch
import matplotlib.pyplot as plt

data = torch.load("data/HOCAP/sequence1/sequence1.pt", map_location="cpu")

print("=" * 50)
print("KEYS AND SHAPES")
print("=" * 50)
for k, v in data.items():
    shape = v.shape if hasattr(v, "shape") else "(scalar)"
    print(f"  {k:20s} {str(shape)}")

print("\n" + "=" * 50)
print("SAMPLE VALUES (first 3 frames)")
print("=" * 50)
print(f"  R_init (quat):       {data['R_init'].numpy()}")
print(f"  t_init (pos):        {data['t_init'].numpy()}")
print(f"  obj_bottom_offset:   {data['obj_bottom_offset'].item():.4f}")

for frame in [0, 1, 2]:
    print(f"\n  --- Frame {frame} ---")
    print(f"  obj_trans:  {data['obj_trans'][frame].numpy()}")
    print(f"  obj_vel:    {data['obj_vel'][frame].numpy()}")
    print(f"  obj_angvel: {data['obj_angvel'][frame].numpy()}")
    print(f"  mano_kpts[wrist]:      {data['mano_kpts'][frame, 0].numpy()}")
    print(f"  mano_kpts[fingertips]: {data['mano_kpts'][frame, [4,8,12,16,20]].numpy()}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Sequence 1 — 250 frames", fontsize=14)

frames = range(250)

# Object trajectory (x, y, z over time)
ax = axes[0, 0]
obj = data["obj_trans"]
for i, label in enumerate(["x", "y", "z"]):
    ax.plot(frames, obj[:, i], label=label)
ax.set_title("Object position over time")
ax.set_xlabel("Frame")
ax.legend()

# Object velocity magnitude
ax = axes[0, 1]
obj_vel_mag = torch.norm(data["obj_vel"], dim=-1)
ax.plot(frames, obj_vel_mag)
ax.set_title("Object linear speed over time")
ax.set_xlabel("Frame")
ax.set_ylabel("m/s")

# Wrist keypoint trajectory
ax = axes[1, 0]
wrist = data["mano_kpts"][:, 0, :]
for i, label in enumerate(["x", "y", "z"]):
    ax.plot(frames, wrist[:, i], label=label)
ax.set_title("Wrist (MANO kpt 0) position over time")
ax.set_xlabel("Frame")
ax.legend()

# Fingertip positions at frame 0 vs frame 125 vs frame 249
ax = axes[1, 1]
fingertips = [4, 8, 12, 16, 20]
names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
for sample_frame, marker in [(0, "o"), (125, "s"), (249, "^")]:
    pts = data["mano_kpts"][sample_frame, fingertips]
    ax.scatter(pts[:, 0], pts[:, 1], label=f"frame {sample_frame}", marker=marker)
    for i, name in enumerate(names):
        ax.annotate(name, (pts[i, 0].item(), pts[i, 1].item()), fontsize=7)
ax.set_title("Fingertip XY positions (3 sample frames)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

plt.tight_layout()
plt.savefig("sequence1_inspection.png", dpi=120)
print("\nPlot saved to sequence1_inspection.png")
plt.show()