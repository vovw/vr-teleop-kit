"""Record LeRobot datasets from bimanual YAM teleop — controlled entirely
from the Quest controllers, no keyboard needed.

In-VR control map (on top of the normal teleop bindings):

    grip (squeeze)     drive the arm (clutch), trigger = gripper
    A / X (hold)       precision scale
    thumbstick click   ramp that arm to its rest pose (nice between episodes)
    right B            start episode  /  save episode (toggle)
    left Y             discard the episode being recorded (re-do it)
    right B + left Y   hold both ~1.5 s while idle → end the session

Teleop keeps driving the robot between episodes (so you can reset the
scene with the arm), but frames are only written while an episode is
open. Episode transitions are logged loudly; watch the terminal or trust
the muscle memory: B ... do the thing ... B.

The loop runs at the dataset fps (default 30), like `lerobot-record`.
The per-joint velocity caps are per-tick, so at 30 fps the arm is slower
than in the 200 Hz examples — raise the caps from the web Settings panel
if the arm feels sluggish while recording.

Requires lerobot (this is the one entry point that genuinely needs it —
on a weak connection install CPU-only torch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

Run (hardware):
    python examples/record_bi_yam.py \\
        --repo-id you/yam-task --task "fold the towel" \\
        --left-can can_left --right-can can_right

Rehearse the whole flow without motors (i2rt sim robots):
    python examples/record_bi_yam.py \\
        --repo-id you/yam-task --task "rehearsal" --sim
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

from vr_teleop_kit.lerobot import init_logging
from vr_teleop_kit.lerobot.bi_quest_teleop import (
    BiQuestTeleoperator,
    BiQuestTeleoperatorConfig,
)
from vr_teleop_kit.lerobot.cli import (
    add_ik_cli_args,
    ik_kwargs_from_args,
    parse_rest_pose_env,
)

# Shares the robot-facing helpers with the plain teleop example.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teleop_bi_yam import ARM_DOFS, _command_arm, _gripper_feedback, ramp_to_rest  # noqa: E402

# Hold right-B + left-Y this long (while idle) to end the session.
END_SESSION_HOLD_S = 1.5


def _obs_vector(hands: tuple[str, ...], robots: dict) -> np.ndarray:
    """Per arm: [joint1..6 (rad), gripper (0=open..1=closed)], left first.
    Gripper converted from i2rt's normalized convention (0=closed)."""
    out = []
    for hand in hands:
        obs = robots[hand].get_observations()
        out.extend(float(v) for v in obs["joint_pos"][:ARM_DOFS])
        out.append(1.0 - float(obs["gripper_pos"][0]))
    return np.asarray(out, dtype=np.float32)


def _action_vector(action: dict[str, float], hands: tuple[str, ...]) -> np.ndarray:
    out = []
    for hand in hands:
        out.extend(float(action[f"{hand}_joint_{j + 1}.pos"]) for j in range(ARM_DOFS))
        out.append(float(action[f"{hand}_gripper.pos"]))
    return np.asarray(out, dtype=np.float32)


def _feature_names(hands: tuple[str, ...]) -> list[str]:
    return [f"{hand}_{n}" for hand in hands
            for n in [f"joint_{j + 1}" for j in range(ARM_DOFS)] + ["gripper"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="dataset repo id (e.g. you/yam-towels)")
    ap.add_argument("--task", required=True, help="task string stored with every frame")
    ap.add_argument("--root", default=None, help="local dataset root (default: LeRobot's)")
    ap.add_argument("--fps", type=int, default=30, help="dataset + control-loop rate")
    ap.add_argument("--num-episodes", type=int, default=0,
                    help="stop after this many saved episodes (0 = until ended from VR)")
    ap.add_argument("--left-can", default="can_left", help="left arm CAN interface")
    ap.add_argument("--right-can", default="can_right", help="right arm CAN interface")
    ap.add_argument("--arm", choices=("both", "left", "right"), default="both")
    ap.add_argument("--sim", action="store_true",
                    help="use i2rt sim robots (rehearse the recording flow, no hardware)")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws", help="relay server WS URL")
    ap.add_argument("--rest-duration-s", type=float, default=3.0)
    ap.add_argument("--rest-steps", type=int, default=90)
    add_ik_cli_args(ap)
    args = ap.parse_args()

    init_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    try:
        from i2rt.robots.get_robot import get_yam_robot
    except ImportError as e:
        raise ImportError(
            "i2rt is required. Clone https://github.com/i2rt-robotics/i2rt "
            "and install it (pip install -e path/to/i2rt)."
        ) from e
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    hands: tuple[str, ...] = ("left", "right") if args.arm == "both" else (args.arm,)
    channels = {"left": args.left_can, "right": args.right_can}

    names = _feature_names(hands)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,
        robot_type="bi_yam" if args.arm == "both" else "yam",
        use_videos=False,
        features={
            "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
            "action": {"dtype": "float32", "shape": (len(names),), "names": names},
        },
    )

    rests = {
        "left": parse_rest_pose_env("LEFT_REST_POSE", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "right": parse_rest_pose_env("RIGHT_REST_POSE", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    robots = {h: get_yam_robot(channel=channels[h], sim=args.sim) for h in hands}

    teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(
        id="vr-record",
        ws_url=args.ws_url,
        rest_qpos_left=rests["left"],
        rest_qpos_right=rests["right"],
        **ik_kwargs_from_args(args),
    ))
    teleop.connect()
    for h in hands:
        ramp_to_rest(robots[h], rests[h], args.rest_duration_s, args.rest_steps, logger, h)

    logger.info("──────────────────────────────────────────────────")
    logger.info("VR recording controls:")
    logger.info("  right B        start / save episode")
    logger.info("  left Y         discard current episode")
    logger.info("  B + Y (hold)   end session (while idle)")
    logger.info("──────────────────────────────────────────────────")

    recording = False
    frames_in_episode = 0
    last_b = last_y = False
    both_held_since: float | None = None
    period = 1.0 / args.fps

    try:
        next_tick = time.perf_counter()
        while True:
            b = teleop.is_pause_pressed()     # right B (level)
            y = teleop.is_reverse_pressed()   # left Y (level)

            # End-session chord: both held while idle.
            if not recording and b and y:
                both_held_since = both_held_since or time.perf_counter()
                if time.perf_counter() - both_held_since >= END_SESSION_HOLD_S:
                    logger.info("B+Y held — ending session")
                    break
            else:
                both_held_since = None

            # Suppress single-button edges while the chord is being formed.
            chording = b and y
            if not chording:
                if b and not last_b:
                    if not recording:
                        recording, frames_in_episode = True, 0
                        logger.info("● episode %d STARTED", dataset.num_episodes)
                    elif frames_in_episode > 0:
                        dataset.save_episode()
                        recording = False
                        logger.info("✓ episode %d SAVED (%d frames)",
                                    dataset.num_episodes - 1, frames_in_episode)
                        if args.num_episodes and dataset.num_episodes >= args.num_episodes:
                            logger.info("reached --num-episodes=%d", args.num_episodes)
                            break
                if y and not last_y and recording:
                    dataset.clear_episode_buffer()
                    recording = False
                    logger.info("✗ episode DISCARDED (%d frames) — go again", frames_in_episode)
            last_b, last_y = b, y

            # lerobot-record ordering: observe, act, command, write.
            obs = _obs_vector(hands, robots)
            action = teleop.get_action()
            torques: dict[str, float] = {}
            for h in hands:
                _command_arm(robots[h], action, f"{h}_")
                try:
                    torques.update(_gripper_feedback(robots[h], f"{h}_"))
                except Exception:
                    pass
            if torques:
                teleop.send_feedback({"torques": torques})

            if recording:
                dataset.add_frame({
                    "observation.state": obs,
                    "action": _action_vector(action, hands),
                    "task": args.task,
                })
                frames_in_episode += 1

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        if recording:
            dataset.clear_episode_buffer()
            logger.info("open episode discarded (session ended mid-episode)")
        if hasattr(dataset, "finalize"):
            dataset.finalize()
        logger.info("session done: %d episode(s) at %s", dataset.num_episodes, dataset.root)
        try:
            teleop.disconnect()
        finally:
            for h in hands:
                robots[h].close()


if __name__ == "__main__":
    main()
