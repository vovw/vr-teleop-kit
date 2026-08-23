"""Capture synchronised camera frames + joint states, for wrist-camera calibration.

Wrist-camera *intrinsics* come off the D405 EEPROM and are already exact. What
cannot be measured from the greenscreen stills is the **extrinsics** -- where
each camera sits on its gripper -- because solving that needs the gripper's
world pose, which needs the joint angles at the instant the shutter fired. The
existing stills carry no joint states, so this tool captures both together.

Several poses, not one: from a single arm pose the mount transform is close to
degenerate (rotation about the camera's own axis is barely observable), and a
one-pose fit will happily report a small residual while being wrong. Move the
arms to a genuinely different configuration between captures -- different
heights and yaws, gripper pointing at different parts of the table.

The arms come up in gravity-compensation mode, so they are back-drivable: pose
them by hand, then press Enter. **This energizes the motors.** The arms are
powered down on exit.

    python tools/sim_capture_calib.py --poses 5

Writes calib/pose_NN/{top,left_wrist,right_wrist}.png and joints.json.
Nothing else may hold the cameras -- stop the relay first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from vr_teleop_kit.relay.capture import CameraReader, build_camera_specs  # noqa: E402

CAMERA_WARMUP_S = 6.0
ARM_DOFS = 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=5,
                   help="how many arm configurations to capture")
    ap.add_argument("--out", default=str(_REPO_ROOT / "calib"))
    ap.add_argument("--left-can", default="can1")
    ap.add_argument("--right-can", default="can0")
    ap.add_argument("--no-arms", action="store_true",
                   help="cameras only -- captures no joint states, so the "
                        "result cannot be used for the extrinsics fit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Cameras first: if a device is busy we want to fail before energizing.
    specs = build_camera_specs()
    if not specs:
        raise SystemExit("no cameras discovered")
    readers = {}
    for spec in specs:
        r = CameraReader(spec)
        deadline = time.time() + CAMERA_WARMUP_S
        while time.time() < deadline and r.latest() is None:
            time.sleep(0.05)
        if r.latest() is None:
            r.stop()
            print(f"camera {spec.id}: no frame -- is the relay holding it?")
            continue
        readers[spec.id] = r
        print(f"camera {spec.id} ready ({spec.label})")
    if not readers:
        raise SystemExit("no camera produced a frame")

    robots = {}
    try:
        if not args.no_arms:
            from i2rt.robots.get_robot import get_yam_robot
            for hand, chan in (("left", args.left_can), ("right", args.right_can)):
                print(f"bringing up {hand} arm on {chan} (motors ON, "
                      f"gravity-comp / back-drivable)")
                robots[hand] = get_yam_robot(channel=chan)

        for i in range(args.poses):
            print(f"\n--- pose {i+1}/{args.poses} ---")
            print("pose the arms by hand, then press Enter (Ctrl-C to stop early)")
            input()
            # Grab joint states and frames as close together as possible.
            stamp = time.time()
            joints = {}
            for hand, robot in robots.items():
                obs = robot.get_observations()
                joints[hand] = {
                    "joint_pos": [float(v) for v in obs["joint_pos"][:ARM_DOFS]],
                    "gripper_pos": float(obs["gripper_pos"][0]),
                    # record the raw vector too, in case the layout matters later
                    "joint_pos_raw": [float(v) for v in obs["joint_pos"]],
                }
            frames = {cid: r.latest() for cid, r in readers.items()}

            pose_dir = out_root / f"pose_{i:02d}"
            pose_dir.mkdir(parents=True, exist_ok=True)
            for cid, frame in frames.items():
                if frame is None:
                    print(f"  {cid}: no frame, skipped")
                    continue
                cv2.imwrite(str(pose_dir / f"{cid}.png"), frame)
            (pose_dir / "joints.json").write_text(json.dumps(
                {"timestamp": stamp, "arms": joints}, indent=2) + "\n")
            for hand, j in joints.items():
                print(f"  {hand}: q=" +
                      ", ".join(f"{v:+.4f}" for v in j["joint_pos"]) +
                      f"  grip={j['gripper_pos']:.3f}")
            print(f"  wrote {pose_dir}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        for r in readers.values():
            r.stop()
        for hand, robot in robots.items():
            print(f"powering down {hand} arm")
            try:
                robot.close()
            except Exception:
                logging.exception("closing %s arm failed", hand)
        print(f"\ncaptures in {out_root}")


if __name__ == "__main__":
    main()
