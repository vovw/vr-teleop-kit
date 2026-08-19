"""Prove the EE follower in sim — no hardware, no relay.

Three checks, in increasing order of what they'd catch:

1. **Frame round-trip.** Drive the follower with a target built by forward
   kinematics from a known joint configuration, in both supported frames, and
   confirm it converges back to that configuration. This is what catches a
   botched tool0/TCP conversion — the failure mode that puts the arm 13.5 cm
   past everything.

2. **Ground-truth tracking.** Replay a recorded episode's end-effector actions
   through the follower and compare its joint output against the
   `commanded_joints` that actually produced those poses. The dataset carries
   both, so the correct answer is known exactly. This is the real question:
   can the deployment path reconstruct a trajectory it never saw the joints
   for?

3. **Policy-like abuse.** Step jumps, per-sample jitter and an unreachable
   target — the ways a policy differs from a human hand. Asserts the solver
   stays finite, respects its per-tick joint cap, and recovers.

Usage:
    python tools/sim_check_ee_follower.py --root /tmp/ee-sim-check

The dataset must come from `record_bi_ee_only.py --with-joints` (check 2 needs
the joint features). Checks 1 and 3 need no data at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from vr_teleop_kit.ik.decoupled_ik import DecoupledIKSolver  # noqa: E402
from vr_teleop_kit.ik.ee_follower import EEFollower  # noqa: E402

ARM_BLOCK = 10  # pos(3) + rot6d(6) + gripper(1)


def _rot6d_to_quat_xyzw(vec6) -> np.ndarray:
    """Gram-Schmidt decode, then to xyzw — mirrors the recorder's encoder."""
    import mujoco

    a = np.asarray(vec6, dtype=np.float64)
    b1 = a[:3] / np.linalg.norm(a[:3])
    b2 = a[3:] - b1.dot(a[3:]) * b1
    b2 /= np.linalg.norm(b2)
    R = np.stack((b1, b2, np.cross(b1, b2)), axis=-2)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(9))
    xyzw = np.array([q[1], q[2], q[3], q[0]])
    return -xyzw if xyzw[3] < 0 else xyzw


def check_frames(mu: float) -> None:
    """A target built by FK from q_goal must drive the follower back to q_goal."""
    solver = DecoupledIKSolver(mu=mu)
    q_goal = np.array([0.35, 0.55, 0.65, 0.15, -0.30, 0.25])

    # The posture-bias term is a steady-state offset against a fixed target.
    # Report it so the deployment value is a measured choice, not a default.
    for probe_mu in (0.02, 0.005, 0.0):
        s_probe = DecoupledIKSolver(mu=probe_mu)
        g = EEFollower(s_probe, freq=200.0, target_frame="tool0", q_init=np.zeros(6))
        g._q = q_goal.copy()
        tp, tq = g.ee_pose()
        g._q = np.zeros(6)
        g.set_target(tp, tq)
        for _ in range(1500):
            g.step()
        note = "   <- teleop default" if probe_mu == 0.02 else ""
        print(f"  mu={probe_mu:<6} steady-state offset "
              f"{np.linalg.norm(g.ee_pose()[0] - tp) * 1000:6.3f} mm{note}")
    print(f"  using mu={mu} for the checks below")

    for frame in ("tool0", "tcp"):
        f = EEFollower(solver, freq=200.0, target_frame=frame,
                       q_init=np.zeros(6), gripper_init=0.0)
        # Build the target in this frame from the goal configuration itself.
        f._q = q_goal.copy()
        pos, quat = f.ee_pose(frame)
        f._q = np.zeros(6)  # start far away

        f.set_target(pos, quat, gripper=1.0)
        for _ in range(1200):  # 6 s at 200 Hz
            f.step()

        reached_pos, reached_quat = f.ee_pose(frame)
        pos_err = float(np.linalg.norm(reached_pos - pos))
        dot = abs(float(np.dot(reached_quat, quat)))
        ang_err = np.degrees(2 * np.arccos(min(1.0, dot)))
        print(f"  frame={frame:6s} converged: pos err {pos_err * 1000:.3f} mm, "
              f"rot err {ang_err:.3f} deg, gripper {f.gripper:.2f}")
        assert pos_err < 1e-3, f"{frame}: position did not converge ({pos_err:.4f} m)"
        assert ang_err < 0.5, f"{frame}: orientation did not converge ({ang_err:.3f} deg)"
        assert f.gripper > 0.99, "gripper did not reach its target"

    # The two frames must genuinely differ, or the conversion is a no-op and
    # check 1 would pass while hiding the 13.5 cm bug.
    f_tool0 = EEFollower(solver, target_frame="tool0", q_init=q_goal)
    f_tcp = EEFollower(solver, target_frame="tcp", q_init=q_goal)
    d = float(np.linalg.norm(f_tool0.ee_pose()[0] - f_tcp.ee_pose()[0]))
    print(f"  tool0 vs tcp separation at the same joints: {d * 100:.2f} cm")
    assert 0.13 < d < 0.14, f"expected ~13.5 cm between the frames, got {d:.4f} m"


def check_tracking(root: Path, repo_id: str, inner_hz: float, policy_hz: float, mu: float) -> None:
    """Replay recorded EE actions; compare against the joints that made them."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    feats = ds.meta.features
    if "commanded_joints" not in feats:
        raise SystemExit("dataset lacks 'commanded_joints' — re-record with --with-joints")
    n = feats["action"]["shape"][0]
    n_arms = n // ARM_BLOCK
    arm = n_arms - 1  # right arm is the last block
    print(f"  {len(ds)} frames, {n_arms} arm(s); replaying the right arm at "
          f"{policy_hz:.0f} Hz targets / {inner_hz:.0f} Hz inner loop")

    action = np.stack([ds[i]["action"].numpy() for i in range(len(ds))])
    cmd_q = np.stack([ds[i]["commanded_joints"].numpy() for i in range(len(ds))])
    j0 = 7 * arm  # per arm: 6 joints + gripper
    a0 = ARM_BLOCK * arm

    solver = DecoupledIKSolver(mu=mu)
    # Seed from the first recorded command, as a real deployment seeds from the
    # arm's measured joints at startup.
    f = EEFollower(solver, freq=inner_hz, target_frame="tool0",
                   q_init=cmd_q[0, j0:j0 + 6], gripper_init=float(cmd_q[0, j0 + 6]))

    steps_per_target = max(1, int(round(inner_hz / policy_hz)))
    joint_err, pos_err, grip_err = [], [], []
    for i in range(len(ds)):
        block = action[i, a0:a0 + ARM_BLOCK]
        f.set_target(block[:3], _rot6d_to_quat_xyzw(block[3:9]), gripper=float(block[9]))
        for _ in range(steps_per_target):
            f.step()
        joint_err.append(np.abs(f.qpos - cmd_q[i, j0:j0 + 6]).max())
        pos_err.append(np.linalg.norm(f.ee_pose("tool0")[0] - block[:3]))
        grip_err.append(abs(f.gripper - float(block[9])))

    joint_err, pos_err = np.array(joint_err), np.array(pos_err)
    # Skip the first few targets: the follower starts at the seed and needs a
    # moment to catch the trajectory, which is expected, not a defect.
    warm = min(10, len(pos_err) - 1)
    print(f"  joint error   : median {np.degrees(np.median(joint_err[warm:])):.3f} deg, "
          f"max {np.degrees(joint_err[warm:].max()):.3f} deg")
    print(f"  EE pos error  : median {np.median(pos_err[warm:]) * 1000:.3f} mm, "
          f"max {pos_err[warm:].max() * 1000:.3f} mm")
    print(f"  gripper error : max {max(grip_err[warm:]):.4f}")
    assert np.isfinite(pos_err).all()
    assert np.median(pos_err[warm:]) < 2e-3, "median EE tracking worse than 2 mm"
    assert pos_err[warm:].max() < 1e-2, "peak EE tracking worse than 10 mm"
    assert np.degrees(joint_err[warm:].max()) < 5.0, "joint trajectory diverged"


def check_abuse(inner_hz: float, mu: float) -> None:
    """Step jumps, jitter and an unreachable target — policy-shaped inputs."""
    solver = DecoupledIKSolver(mu=mu)
    q_start = np.array([0.2, 0.5, 0.6, 0.0, -0.2, 0.0])
    f = EEFollower(solver, freq=inner_hz, target_frame="tool0", q_init=q_start)
    cap = getattr(solver, "max_dq_per_joint", None)

    # A 15 cm instantaneous jump — far larger than any single teleop tick.
    base_pos, base_quat = f.ee_pose()
    f.set_target(base_pos + np.array([0.0, 0.0, 0.15]), base_quat, gripper=1.0)
    worst_dq = 0.0
    for _ in range(int(inner_hz * 3)):
        prev = f.qpos
        f.step()
        worst_dq = max(worst_dq, float(np.abs(f.qpos - prev).max()))
    jump_err = float(np.linalg.norm(f.ee_pose()[0] - (base_pos + np.array([0.0, 0.0, 0.15]))))
    print(f"  15 cm step jump: settled to {jump_err * 1000:.2f} mm, "
          f"worst per-tick joint delta {worst_dq:.4f} rad")
    assert np.isfinite(f.qpos).all()
    if cap is not None:
        assert worst_dq <= float(np.max(cap)) + 1e-9, "per-tick joint cap violated"

    # Per-sample jitter around a fixed pose, as a noisy policy would emit.
    rng = np.random.default_rng(0)
    hold_pos, hold_quat = f.ee_pose()
    for _ in range(int(inner_hz)):
        f.set_target(hold_pos + rng.normal(scale=0.003, size=3), hold_quat, gripper=0.0)
        f.step()
    jitter_err = float(np.linalg.norm(f.ee_pose()[0] - hold_pos))
    print(f"  3 mm jitter    : stayed within {jitter_err * 1000:.2f} mm of the pose")
    assert np.isfinite(f.qpos).all() and jitter_err < 0.02

    # Unreachable: 3 m away. Must degrade, not explode or NaN.
    f.set_target(np.array([3.0, 0.0, 0.5]), hold_quat, gripper=0.0)
    for _ in range(int(inner_hz * 2)):
        f.step()
    assert np.isfinite(f.qpos).all(), "unreachable target produced non-finite joints"
    reach = f.qpos.copy()
    print(f"  unreachable 3 m: finite joints, max |q| = {np.abs(reach).max():.3f} rad")

    # ...and recovers when the target becomes reachable again.
    f.set_target(hold_pos, hold_quat, gripper=0.0)
    for _ in range(int(inner_hz * 4)):
        f.step()
    recover = float(np.linalg.norm(f.ee_pose()[0] - hold_pos))
    print(f"  recovery       : back to {recover * 1000:.2f} mm")
    assert recover < 5e-3, "did not recover after an unreachable target"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/ee-sim-check",
                    help="LeRobot dataset root from record_bi_ee_only.py --with-joints")
    ap.add_argument("--repo-id", default="local/ee-sim-check")
    ap.add_argument("--inner-hz", type=float, default=200.0, help="follower loop rate")
    ap.add_argument("--policy-hz", type=float, default=30.0,
                    help="rate targets are handed to the follower (the dataset fps)")
    ap.add_argument("--mu", type=float, default=0.0,
                    help="IK posture-bias term for the rollout solver. The teleop "
                         "default (0.02) leaves a 3.4 mm steady-state offset against "
                         "a fixed target; deployment wants this near 0 (default: 0.0)")
    ap.add_argument("--skip-tracking", action="store_true",
                    help="run only the checks that need no dataset")
    args = ap.parse_args()

    print("1. frame round-trip (tool0 and tcp):")
    check_frames(args.mu)

    if not args.skip_tracking:
        print("\n2. ground-truth tracking against recorded commanded_joints:")
        check_tracking(Path(args.root), args.repo_id, args.inner_hz, args.policy_hz, args.mu)

    print("\n3. policy-like abuse:")
    check_abuse(args.inner_hz, args.mu)

    print("\nEE FOLLOWER CHECKS PASSED")


if __name__ == "__main__":
    main()
