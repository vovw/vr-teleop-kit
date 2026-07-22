"""Direct gripper open/close probe for a single YAM arm — no VR, no teleop.

Isolates whether a gripper that "closes but won't open" is a hardware /
i2rt-calibration fault or something in the VR path. Commands the gripper
through open → closed → open in slow steps while printing the commanded
value alongside the measured position + motor effort.

i2rt gripper convention (see examples/teleop_bi_yam.py): the 7th command
element is 0 = closed, 1 = open. get_observations()["gripper_pos"] reads
back in the same 0..1 scale; "gripper_eff" is motor effort in Nm.

Usage:
    python tools/gripper_probe.py --can can_right
    python tools/gripper_probe.py --can can_left   # compare the good arm

Watch for:
  * gripper_pos tracks the commanded value on both arms  -> healthy
  * commanded 1.0 (open) but pos stays near 0 (closed)   -> won't open
  * gripper_eff pinned high / climbing while stuck        -> stalled or
    faulted motor (overcurrent from slamming a hard stop); power-cycle
    the arm to clear a latched fault, then recalibrate the gripper range.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from i2rt.robots.get_robot import get_yam_robot


def read(robot) -> tuple[float | None, float | None]:
    obs = robot.get_observations()
    pos = float(obs["gripper_pos"][0]) if "gripper_pos" in obs else None
    eff = float(obs["gripper_eff"][0]) if "gripper_eff" in obs else None
    return pos, eff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--can", required=True, help="CAN channel, e.g. can_right")
    ap.add_argument("--hold-s", type=float, default=1.5,
                    help="seconds to hold at each end before reporting")
    ap.add_argument("--steps", type=int, default=20,
                    help="interpolation steps between open and closed")
    args = ap.parse_args()

    robot = get_yam_robot(channel=args.can)
    try:
        # Freeze the 6 arm joints at their current pose; only move the gripper.
        arm = np.asarray(robot.get_joint_pos(), dtype=float)[:6].copy()

        def command(grip: float) -> None:
            robot.command_joint_pos(np.concatenate([arm, [grip]]))

        # open (1.0) -> closed (0.0) -> open (1.0)
        for label, lo, hi in (("CLOSE", 1.0, 0.0), ("OPEN", 0.0, 1.0)):
            print(f"\n=== {label}: commanding {lo:.2f} -> {hi:.2f} ===")
            for i in range(args.steps + 1):
                g = lo + (hi - lo) * (i / args.steps)
                command(g)
                time.sleep(args.hold_s / args.steps)
                pos, eff = read(robot)
                print(f"  cmd={g:5.2f}  pos={pos!s:>7}  eff_Nm={eff!s:>8}")
            time.sleep(args.hold_s)
            pos, eff = read(robot)
            print(f"  settled: pos={pos}  eff_Nm={eff}")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
