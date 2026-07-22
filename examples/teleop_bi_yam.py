"""VR bimanual teleop — live hardware example (two I2RT YAM arms).

Runs the BiQuestTeleoperator against a pair of YAM arms driven through
i2rt, alongside the relay server (`vr-teleop-relay`) and the WebXR
client on the Quest.

At startup, the teleop's internal qpos is initialised to the rest pose
(LEFT_REST_POSE / RIGHT_REST_POSE env vars, seven comma-separated values
with the trailing gripper dropped; all-zeros default if unset), and
`ramp_to_rest()` drives the physical arms there in linearly-interpolated
steps (--rest-duration-s / --rest-steps). The operator can then squeeze
grip to engage and start teleoperating from the same anchor pose.

Pre-requisites:
  - Relay server + transport up (`vr-teleop-relay`; see the README for the
    USB `adb reverse` and LAN HTTPS transports)
  - Quest browser at http://localhost:8443/ (USB) or
    https://<workstation-lan-ip>:8443/ (LAN), in passthrough VR
  - i2rt installed from a clone of https://github.com/i2rt-robotics/i2rt
    (pip install -e path/to/i2rt); the same clone provides the YAM model
    files the IK loads (see YAM_XML in the README)
  - One CAN interface per arm (e.g. can_left / can_right), each with a
    YAM + linear_4310 gripper chain on it

Run:
    python examples/teleop_bi_yam.py \\
        --left-can can_left \\
        --right-can can_right
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from vr_teleop_kit.lerobot.bi_quest_teleop import (
    BiQuestTeleoperator,
    BiQuestTeleoperatorConfig,
)
from vr_teleop_kit.lerobot.cli import (
    add_ik_cli_args,
    ik_kwargs_from_args,
    parse_rest_pose_env,
)

from vr_teleop_kit.log import get_logger, setup_logging

ARM_DOFS = 6


def _command_arm(robot, action: dict[str, float], prefix: str) -> None:
    """Send one arm's slice of the teleop action dict to an i2rt YAM.

    i2rt expects a 7-vector [joint1..joint6, gripper] with the gripper
    normalized so 0 = closed and 1 = open (JointMapper maps 0..1 onto the
    calibrated [closed, open] motor range) — the inverse of the teleop's
    `gripper.pos` convention (0 = open, 1 = closed).
    """
    cmd = np.empty(ARM_DOFS + 1)
    for j in range(ARM_DOFS):
        cmd[j] = action[f"{prefix}joint_{j + 1}.pos"]
    cmd[ARM_DOFS] = 1.0 - float(np.clip(action[f"{prefix}gripper.pos"], 0.0, 1.0))
    robot.command_joint_pos(cmd)


def _gripper_feedback(robot, prefix: str) -> dict[str, float]:
    """Best-effort gripper torque + position readout for the force-haptic
    path (`teleop.send_feedback`). Positions are converted back to the
    teleop's 0=open..1=closed convention; the torque is the raw gripper
    motor effort in Nm (deadband it via the web UI's Calibrate button)."""
    obs = robot.get_observations()
    out: dict[str, float] = {}
    if "gripper_eff" in obs:
        out[f"{prefix}gripper.torque"] = float(obs["gripper_eff"][0])
    if "gripper_pos" in obs:
        out[f"{prefix}gripper.pos"] = 1.0 - float(obs["gripper_pos"][0])
    return out


def ramp_to_rest(
    robot,
    rest_joints: list[float],
    duration_s: float,
    steps: int,
    logger: logging.Logger,
    label: str,
) -> None:
    """Smoothly drive one YAM from its current pose to the rest pose (with
    the gripper open) over `duration_s` seconds in `steps` increments."""
    target = np.array(list(rest_joints) + [1.0])  # i2rt gripper: 1 = open
    start = np.asarray(robot.get_joint_pos(), dtype=float).copy()

    logger.info("ramping to rest pose over %.1fs in %d steps", duration_s, steps)
    dt = duration_s / max(1, steps)
    for i in range(1, steps + 1):
        alpha = i / steps
        robot.command_joint_pos(start + alpha * (target - start))
        time.sleep(dt)
    logger.info("rest pose reached")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-can", default="can1",
                    help="left arm CAN interface (default: can1)")
    ap.add_argument("--right-can", default="can0",
                    help="right arm CAN interface (default: can0)")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws", help="relay server WS URL")
    ap.add_argument("--freq", type=int, default=200, help="teleop loop rate (Hz)")
    ap.add_argument("--rest-duration-s", type=float, default=3.0,
                    help="seconds to ramp arms to rest pose at startup")
    ap.add_argument("--rest-steps", type=int, default=90,
                    help="number of interpolation steps in the startup ramp")
    ap.add_argument("--arm", choices=("both", "left", "right"), default="both",
                    help="run bimanual ('both') or just one arm. With 'left' or "
                         "'right' the BiQuestTeleoperator still tracks both, but "
                         "only the chosen arm is connected and driven; the other "
                         "controller's IK is computed and ignored.")
    add_ik_cli_args(ap)
    args = ap.parse_args()

    setup_logging(level=logging.INFO)
    logger = get_logger(__name__, "session")

    try:
        from i2rt.robots.get_robot import get_yam_robot
    except ImportError as e:
        raise ImportError(
            "i2rt is required to drive YAM hardware. Clone "
            "https://github.com/i2rt-robotics/i2rt and install it "
            "(pip install -e path/to/i2rt)."
        ) from e

    fallback = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rest_left = parse_rest_pose_env("LEFT_REST_POSE", fallback)
    rest_right = parse_rest_pose_env("RIGHT_REST_POSE", fallback)
    ik_overrides = ik_kwargs_from_args(args)
    logger.info("======== vr-teleop config ========")
    logger.info("arm          : %s", args.arm)
    logger.info("CAN channels : left=%s  right=%s", args.left_can, args.right_can)
    logger.info("rest         : left=%s  right=%s",
                [round(x, 3) for x in rest_left],
                [round(x, 3) for x in rest_right])
    if ik_overrides:
        logger.info("CLI IK overrides: %s", ik_overrides)
    logger.info("==================================")

    # zero_gravity_mode starts the chains in gravity-comp idle; the first
    # command_joint_pos (the rest ramp below) switches them to PD control.
    hands = ("left", "right") if args.arm == "both" else (args.arm,)
    channels = {"left": args.left_can, "right": args.right_can}
    rests = {"left": rest_left, "right": rest_right}
    robots = {hand: get_yam_robot(channel=channels[hand]) for hand in hands}

    teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(
        id="vr-teleop",
        ws_url=args.ws_url,
        rest_qpos_left=rest_left,
        rest_qpos_right=rest_right,
        **ik_overrides,
    ))

    teleop.connect()

    arm_logs = {hand: get_logger(f"{__name__}.{hand}", f"arm{hand}") for hand in hands}
    for hand in hands:
        ramp_to_rest(robots[hand], rests[hand], args.rest_duration_s,
                     args.rest_steps, arm_logs[hand], hand)

    period = 1.0 / args.freq
    logger.info("vr-teleop running at %d Hz (arm=%s); Ctrl-C to stop", args.freq, args.arm)
    try:
        next_tick = time.perf_counter()
        while True:
            action = teleop.get_action()
            torques: dict[str, float] = {}
            for hand in hands:
                _command_arm(robots[hand], action, f"{hand}_")
                # Per-tick force haptic: read gripper torque and forward to
                # the teleop. Best-effort; skipped if the robot doesn't
                # expose it.
                try:
                    torques.update(_gripper_feedback(robots[hand], f"{hand}_"))
                except Exception:
                    pass
            if torques:
                teleop.send_feedback({"torques": torques})
            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        try:
            teleop.disconnect()
        finally:
            for hand in hands:
                robots[hand].close()


if __name__ == "__main__":
    main()
