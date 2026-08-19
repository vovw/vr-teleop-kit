"""No-hardware check for `examples/record_bi_ee_only.py`.

Two modes, neither of which touches a robot, a CAN bus, or a camera:

  --self-test   Pure checks: the rotation encodings, the feature-name/vector
                agreement, the i2rt gripper inversion, FK-instance isolation,
                and a throwaway LeRobot dataset written and read back to
                confirm the schema. Needs nothing running.

  (default)     Drives a full episode against a live relay + a recorder
                started with `--sim`, by acting as a fake Quest: right B to
                start, three seconds of clutched motion on the right hand,
                left Y to save. Then waits for the dataset to finalize and
                verifies its contents — that the right EE actually moved,
                that the unclutched left arm did not, that the gripper
                tracked the trigger, that every rot6d block is a proper
                rotation, and that state/action are exactly FK of the
                measured/commanded joints.

    # terminal 1
    vr-teleop-relay
    # terminal 2
    python examples/record_bi_ee_only.py --sim --no-cameras --with-joints \\
        --num-episodes 1 --repo-id local/ee-sim-check --root /tmp/ee-sim-check
    # terminal 3
    python tools/sim_check_ee_recorder.py --root /tmp/ee-sim-check

The live mode needs `--with-joints` on the recorder: the FK cross-check
compares the recorded EE features against the recorded joint vectors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "examples"))
os.chdir(_REPO_ROOT)

import record_bi_ee_only as rec  # noqa: E402

HZ = 60
ARM_BLOCK = 10  # pos(3) + rot6d(6) + gripper(1)


# --------------------------------------------------------------------------
# fake Quest client
# --------------------------------------------------------------------------

def _buttons(grip=False, trigger=0.0, handoff=False):
    """Seven-entry button array in the WebXR layout the client sends.
    Indices match bi_quest_teleop: 0 trigger, 1 grip, 5 B (right) / Y (left)."""
    b = [{"p": False, "v": 0.0} for _ in range(7)]
    b[0] = {"p": trigger > 0.5, "v": float(trigger)}
    b[1] = {"p": bool(grip), "v": 1.0 if grip else 0.0}
    b[5] = {"p": bool(handoff), "v": 1.0 if handoff else 0.0}
    return b


def _frame(right_pos, right_grip=False, right_trigger=0.0, right_b=False, left_y=False):
    return {
        "type": "xr_frame",
        "t_client": time.time(),
        "controllers": {
            "right": {"position": list(right_pos), "orientation": [0.0, 0.0, 0.0, 1.0],
                      "buttons": _buttons(right_grip, right_trigger, right_b), "axes": []},
            "left": {"position": [-0.2, 1.4, -0.3], "orientation": [0.0, 0.0, 0.0, 1.0],
                     "buttons": _buttons(handoff=left_y), "axes": []},
        },
        "viewer": {"position": [0.0, 1.5, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
    }


async def _drive_episode(ws_url: str, wait_s: float, move_s: float) -> None:
    import websockets

    async with websockets.connect(ws_url) as ws:
        async def send_for(seconds, make_frame):
            for i in range(max(1, int(seconds * HZ))):
                await ws.send(json.dumps(make_frame(i / HZ)))
                await asyncio.sleep(1.0 / HZ)

        base = [0.2, 1.4, -0.3]
        print(f"  idle {wait_s:.0f}s (letting the recorder finish booting)...", flush=True)
        await send_for(wait_s, lambda t: _frame(base))

        print("  right B  -> start episode", flush=True)
        await send_for(0.3, lambda t: _frame(base, right_b=True))
        await send_for(0.3, lambda t: _frame(base))

        print(f"  clutch + move {move_s:.0f}s", flush=True)

        def moving(t):
            pos = [base[0] + 0.06 * math.sin(math.pi * t),
                   base[1] + 0.04 * t,
                   base[2] - 0.03 * t]
            return _frame(pos, right_grip=True, right_trigger=min(1.0, 0.3 * t))

        await send_for(move_s, moving)

        print("  left Y   -> save episode", flush=True)
        await send_for(0.4, lambda t: _frame(base, left_y=True))
        await send_for(1.0, lambda t: _frame(base))


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def _decode_rot6d(vec6) -> np.ndarray:
    """Gram-Schmidt decoder, verbatim from the bspline/diffusion-policy stack."""
    a1, a2 = np.asarray(vec6, dtype=np.float64)[:3], np.asarray(vec6, dtype=np.float64)[3:]
    b1 = a1 / max(np.linalg.norm(a1), 1e-12)
    b2 = a2 - b1.dot(a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-12)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-2)


def self_test() -> None:
    rng = np.random.default_rng(0)

    worst = 0.0
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        worst = max(worst, float(np.abs(
            rec._mat_from_quat_wxyz(q) - _decode_rot6d(rec._rot6d(q))).max()))
    assert worst < 1e-9, f"rot6d convention mismatch ({worst:.2e})"
    print(f"  rot6d round-trips through the training-stack decoder (max ΔR {worst:.1e})")

    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        xyzw = rec._quat_xyzw(q)
        assert xyzw[3] >= 0.0
        back = rec._mat_from_quat_wxyz(np.array([xyzw[3], *xyzw[:3]]))
        assert np.abs(rec._mat_from_quat_wxyz(q) - back).max() < 1e-9
    print("  quat encoding is xyzw with the w >= 0 hemisphere enforced")

    fk = rec.DecoupledIKSolver()
    for rot, (enc, rot_names) in rec.ROTATION_ENCODINGS.items():
        for hands in (("left", "right"), ("right",)):
            names = rec._feature_names(hands, rot_names)
            vec = np.concatenate([np.asarray(rec._ee_vector(fk, np.zeros(6), 0.0, enc))
                                  for _ in hands])
            assert len(names) == len(vec), (rot, hands, len(names), len(vec))
            print(f"  {rot:5s} {'/'.join(hands):11s} -> {len(vec):2d}-dim, names agree")

    # i2rt reports 0 = closed; the dataset must store 0 = open.
    class StubRobot:
        def __init__(self, q, grip):
            self.q, self.g = np.asarray(q, float), float(grip)

        def get_observations(self):
            return {"joint_pos": np.concatenate([self.q, [self.g]]),
                    "gripper_pos": np.array([self.g])}

    hands = ("left", "right")
    enc = rec.ROTATION_ENCODINGS["rot6d"][0]
    measured = rec._measure(hands, {"left": StubRobot(np.zeros(6), 1.0),
                                    "right": StubRobot(np.full(6, 0.2), 0.0)})
    ov = rec._obs_ee_vector(hands, measured, fk, enc)
    jv = rec._obs_joint_vector(hands, measured)
    assert ov.dtype == np.float32 and ov.shape == (20,)
    assert (ov[9], ov[19]) == (0.0, 1.0), "i2rt gripper convention not inverted"
    assert jv.shape == (14,) and (jv[6], jv[13]) == (0.0, 1.0)
    print("  gripper inverted out of i2rt's convention (0 = open in the dataset)")

    # The recorder's FK instance must not disturb another solver's cached anchor.
    other = rec.DecoupledIKSolver()
    other.fk(np.array([0.3, -0.2, 0.5, 0.1, 0.0, 0.0]))
    anchor = other.j4_anchor_xpos().copy()
    fk.fk(np.full(6, 0.9))
    assert np.allclose(other.j4_anchor_xpos(), anchor), "solver state leaked between instances"
    print("  the recorder's FK instance leaves the teleop solvers untouched")

    # Round-trip the exact feature schema through LeRobot.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import dataset_to_policy_features

    # create() requires a non-existent root, so point it at a child of the
    # temp dir rather than the temp dir itself.
    tmp = Path(tempfile.mkdtemp(prefix="ee-schema-"))
    try:
        ds_root = tmp / "ds"
        names = rec._feature_names(hands, rec.ROTATION_ENCODINGS["rot6d"][1])
        joint_names = rec._joint_feature_names(hands)
        ds = LeRobotDataset.create(
            repo_id="local/ee-schema-check", fps=30, root=ds_root,
            robot_type="bi_yam_ee", use_videos=False,
            features={
                "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
                "action": {"dtype": "float32", "shape": (len(names),), "names": names},
                "measured_joints": {"dtype": "float32", "shape": (len(joint_names),), "names": joint_names},
                "commanded_joints": {"dtype": "float32", "shape": (len(joint_names),), "names": joint_names},
            },
        )
        action = {f"{h}_joint_{j + 1}.pos": 0.05 * j for h in hands for j in range(6)}
        action.update({f"{h}_gripper.pos": 0.5 for h in hands})
        for _ in range(4):
            ds.add_frame({
                "observation.state": ov,
                "action": rec._action_ee_vector(action, hands, fk, enc),
                "measured_joints": jv,
                "commanded_joints": rec._joint_action_vector(action, hands),
                "task": "schema check",
            })
        ds.save_episode()
        ds.finalize()
        back = LeRobotDataset("local/ee-schema-check", root=ds_root)
        assert len(back) == 4
        assert tuple(back[0]["observation.state"].shape) == (20,)
        pf = dataset_to_policy_features(back.meta.features)
        assert set(pf) == {"observation.state", "action"}, set(pf)
        print("  LeRobot schema writes + reads back; policy features are EE-only "
              "(joint keys ignored)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _wait_for_dataset(repo_id: str, root: Path, timeout_s: float):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            ds = LeRobotDataset(repo_id, root=root)
            if len(ds) > 0:
                return ds
        except Exception as e:  # still being written / not finalized yet
            last_err = e
        time.sleep(1.0)
    raise TimeoutError(
        f"no finalized episode at {root} after {timeout_s:.0f}s "
        f"(is the recorder running with --root {root}?). Last error: {last_err}")


def verify_dataset(repo_id: str, root: Path, timeout_s: float) -> None:
    ds = _wait_for_dataset(repo_id, root, timeout_s)
    print(f"  {ds.num_episodes} episode(s), {len(ds)} frames @ {ds.fps} fps, "
          f"robot_type={ds.meta.robot_type}")

    feats = ds.meta.features
    n = feats["observation.state"]["shape"][0]
    hands = ("left", "right") if n == 2 * ARM_BLOCK else ("right",)
    assert feats["action"]["shape"][0] == n
    print(f"  state/action {n}-dim, names[0]={feats['observation.state']['names'][0]}")
    if "measured_joints" not in feats:
        raise SystemExit("dataset has no joint features — rerun the recorder with --with-joints")

    state = np.stack([ds[i]["observation.state"].numpy() for i in range(len(ds))])
    action = np.stack([ds[i]["action"].numpy() for i in range(len(ds))])
    meas_q = np.stack([ds[i]["measured_joints"].numpy() for i in range(len(ds))])
    cmd_q = np.stack([ds[i]["commanded_joints"].numpy() for i in range(len(ds))])

    r0 = ARM_BLOCK * (len(hands) - 1)  # right-arm block offset
    travel = float(np.linalg.norm(state[-1, r0:r0 + 3] - state[0, r0:r0 + 3]))
    assert travel > 0.01, f"right EE never moved ({travel:.4f} m)"
    print(f"  right EE travelled {travel:.3f} m under the clutch")

    if len(hands) == 2:
        idle = float(np.linalg.norm(state[-1, 0:3] - state[0, 0:3]))
        assert idle < 1e-6, f"unclutched left arm moved ({idle:.4f} m)"
        print("  unclutched left arm stayed exactly put")

    grip_max = float(action[:, r0 + ARM_BLOCK - 1].max())
    assert grip_max > 0.5, f"trigger ramp not reflected in the gripper action ({grip_max:.2f})"
    print(f"  gripper action tracked the trigger (peak {grip_max:.2f}, 0=open 1=closed)")

    if feats["observation.state"]["names"][3].endswith("rot6d_1"):
        for i in (0, len(ds) // 2, len(ds) - 1):
            for arm in range(len(hands)):
                det = np.linalg.det(_decode_rot6d(action[i, arm * ARM_BLOCK + 3:arm * ARM_BLOCK + 9]))
                assert abs(det - 1.0) < 1e-6, f"frame {i}: rot6d is not a rotation (det={det})"
        print("  rot6d blocks decode to proper rotations (det=+1)")

    fk = rec.DecoupledIKSolver()
    enc = rec.ROTATION_ENCODINGS["rot6d" if "rot6d_1" in feats["observation.state"]["names"][3]
                                else "quat"][0]
    worst_a = worst_s = 0.0
    for i in range(len(ds)):
        exp_a, exp_s = [], []
        for arm in range(len(hands)):
            j = 7 * arm
            exp_a.extend(rec._ee_vector(fk, cmd_q[i, j:j + 6], cmd_q[i, j + 6], enc))
            exp_s.extend(rec._ee_vector(fk, meas_q[i, j:j + 6], meas_q[i, j + 6], enc))
        worst_a = max(worst_a, float(np.abs(np.asarray(exp_a) - action[i]).max()))
        worst_s = max(worst_s, float(np.abs(np.asarray(exp_s) - state[i]).max()))
    assert worst_a < 1e-6, f"action is not FK of the commanded joints ({worst_a:.2e})"
    assert worst_s < 1e-6, f"state is not FK of the measured joints ({worst_s:.2e})"
    print(f"  action == FK(commanded joints) to {worst_a:.1e}, "
          f"state == FK(measured joints) to {worst_s:.1e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the pure checks only (nothing else needs to be running)")
    ap.add_argument("--root", default="/tmp/ee-sim-check",
                    help="dataset root the recorder was started with")
    ap.add_argument("--repo-id", default="local/ee-sim-check")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws")
    ap.add_argument("--wait", type=float, default=9.0,
                    help="seconds of idle frames before pressing B, so the recorder "
                         "has finished loading the model and ramping (default: 9)")
    ap.add_argument("--move-s", type=float, default=3.0, help="seconds of clutched motion")
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="seconds to wait for the dataset to finalize")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip the fake-Quest drive, just verify an existing dataset")
    args = ap.parse_args()

    if args.self_test:
        print("self-test (no relay, no robot, no dataset root needed):")
        self_test()
        print("\nSELF-TEST PASSED")
        return

    root = Path(args.root)
    if not args.verify_only:
        print(f"driving a fake Quest episode at {args.ws_url}:")
        asyncio.run(_drive_episode(args.ws_url, args.wait, args.move_s))
    print(f"\nverifying {args.repo_id} at {root}:")
    verify_dataset(args.repo_id, root, args.timeout)
    print("\nSIM CHECK PASSED")


if __name__ == "__main__":
    main()
