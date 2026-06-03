"""
Reward function sanity tests.
Run from the project root:
    python test_reward.py

No IsaacLab needed — pure torch.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source", "gr"))
from gr.tasks.direct.gr.gr_env import compute_rewards

N = 4  # fake batch of 4 envs

def make_batch(n=N):
    quat = torch.zeros(n, 4); quat[:, 0] = 1.0  # identity quaternion
    return dict(
        obj_pos      = torch.zeros(n, 3),
        obj_pos_ref  = torch.zeros(n, 3),
        obj_rot      = quat.clone(),
        obj_rot_ref  = quat.clone(),
        fingertip_pos     = torch.zeros(n, 5, 3),
        fingertip_pos_ref = torch.zeros(n, 5, 3),
        actions      = torch.zeros(n, 27),
        hand_dof_vel = torch.zeros(n, 22),
    )

# ── TEST 1: perfect tracking → reward should be ~3.0 ──────────────────────────
print("\n── TEST 1: perfect tracking (zero error) ──")
b = make_batch()
reward, logs = compute_rewards(
    b["obj_pos"], b["obj_pos_ref"],
    b["obj_rot"], b["obj_rot_ref"],
    b["fingertip_pos"], b["fingertip_pos_ref"],
    b["actions"], b["hand_dof_vel"],
    action_penalty_scale=-0.004,
    dof_penalty_scale=-0.001,
)
print(f"  reward/total    : {reward[0].item():.4f}  (expected ~3.0)")
print(f"  reward/obj_pos  : {logs['reward/obj_pos'][0].item():.4f}  (expected 1.0)")
print(f"  reward/obj_rot  : {logs['reward/obj_rot'][0].item():.4f}  (expected 1.0)")
print(f"  reward/fingertip: {logs['reward/fingertip'][0].item():.4f}  (expected 1.0)")
assert reward[0].item() > 2.9, "FAIL: perfect tracking should give ~3.0"
print("  PASS")

# ── TEST 2: large error → reward should be ~0.0 ───────────────────────────────
print("\n── TEST 2: large error (0.5m off) ──")
b = make_batch()
b["obj_pos"][:] = 0.5  # 0.5m away
b["fingertip_pos"][:] = 0.5
reward, logs = compute_rewards(
    b["obj_pos"], b["obj_pos_ref"],
    b["obj_rot"], b["obj_rot_ref"],
    b["fingertip_pos"], b["fingertip_pos_ref"],
    b["actions"], b["hand_dof_vel"],
    action_penalty_scale=-0.004,
    dof_penalty_scale=-0.001,
)
print(f"  reward/total    : {reward[0].item():.4f}  (expected near 0)")
print(f"  reward/obj_pos  : {logs['reward/obj_pos'][0].item():.4f}")
print(f"  reward/fingertip: {logs['reward/fingertip'][0].item():.4f}")
assert reward[0].item() < 0.5, "FAIL: large error should give near-zero reward"
print("  PASS")

# ── TEST 3: gradient direction — moving closer should increase reward ──────────
print("\n── TEST 3: gradient direction (closer = higher reward) ──")
errors = [0.3, 0.2, 0.1, 0.05, 0.0]
rewards = []
for e in errors:
    b = make_batch()
    b["obj_pos"][:, 0] = e
    r, _ = compute_rewards(
        b["obj_pos"], b["obj_pos_ref"],
        b["obj_rot"], b["obj_rot_ref"],
        b["fingertip_pos"], b["fingertip_pos_ref"],
        b["actions"], b["hand_dof_vel"],
        action_penalty_scale=0.0, dof_penalty_scale=0.0,
    )
    rewards.append(r[0].item())
    print(f"  err={e:.2f}m → reward={r[0].item():.4f}")

assert all(rewards[i] < rewards[i+1] for i in range(len(rewards)-1)), \
    "FAIL: reward should strictly increase as error decreases"
print("  PASS: reward increases monotonically as error decreases")

# ── TEST 4: rotation reward — 90° off vs aligned ──────────────────────────────
print("\n── TEST 4: rotation reward ──")
b = make_batch()
# 90° rotation around Z axis
b["obj_rot"][:] = torch.tensor([0.7071, 0.0, 0.0, 0.7071])  # 90° off
r_bad, logs_bad = compute_rewards(
    b["obj_pos"], b["obj_pos_ref"],
    b["obj_rot"], b["obj_rot_ref"],
    b["fingertip_pos"], b["fingertip_pos_ref"],
    b["actions"], b["hand_dof_vel"],
    action_penalty_scale=0.0, dof_penalty_scale=0.0,
)
print(f"  90° off  → obj_rot_reward: {logs_bad['reward/obj_rot'][0].item():.4f}  (expected < 0.5)")
b2 = make_batch()
r_good, logs_good = compute_rewards(
    b2["obj_pos"], b2["obj_pos_ref"],
    b2["obj_rot"], b2["obj_rot_ref"],
    b2["fingertip_pos"], b2["fingertip_pos_ref"],
    b2["actions"], b2["hand_dof_vel"],
    action_penalty_scale=0.0, dof_penalty_scale=0.0,
)
print(f"  aligned  → obj_rot_reward: {logs_good['reward/obj_rot'][0].item():.4f}  (expected 1.0)")
assert logs_bad["reward/obj_rot"][0].item() < logs_good["reward/obj_rot"][0].item(), \
    "FAIL: aligned rotation should give higher reward"
print("  PASS")

print("\n══ ALL TESTS PASSED — reward function is logically correct ══\n")
