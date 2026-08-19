"""Record ONE teleop session into TWO datasets at once.

A single right-arm YAM session driven from the Quest produces:

  1. **LeRobot v3.0** — 7-DoF joint space (`joint_1..6` + gripper), with the
     top and wrist cameras as mp4. What `record_bi_yam.py --arm right` writes.
  2. **bspline/tidybot2** — end-effector only (`arm_pos`, `arm_quat`,
     `gripper_pos`), with the same two camera feeds, in the per-episode
     `data.pkl` + `<key>.mp4` layout their `EpisodeReader` consumes.

Both are written from the same control tick, so episode *N* is the same take
in each and the two stay in lockstep: right B starts both, left Y saves both,
a discard drops both. The session ends after `--num-episodes` saved episodes.

Rates. The arm is commanded at `--control-freq` (200 Hz, the same as the plain
teleop example, which is what makes it feel smooth); each dataset is *sampled*
from that loop at its own rate. Commanding at the dataset rate instead is what
made an earlier version jittery — the arm got a new joint target only every
1/fps s and the IK's per-tick velocity cap tightened by the same factor. Each
dataset rate must divide `--control-freq` exactly, so frame timestamps stay
honest rather than drifting against the declared fps.

Frames are recorded uncropped at the camera's native resolution. Crop later,
in the conversion/preprocessing step: same benefit to the policy, but you can
retune the rectangle without re-recording. Whatever crop you choose must then
be applied identically at deployment.

Run (hardware, right arm, 3 episodes):
    python examples/record_dual_yam.py --num-episodes 3 \\
        --repo-id you/yam-dual --bspline-dir data/demos-ee

Rehearse with no motors and no cameras:
    python examples/record_dual_yam.py --sim --no-cameras --num-episodes 1 \\
        --repo-id local/dual-rehearsal --overwrite
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from vr_teleop_kit.log import get_logger, setup_logging
from vr_teleop_kit.lerobot.bi_quest_teleop import (
    BiQuestTeleoperator,
    BiQuestTeleoperatorConfig,
)
from vr_teleop_kit.lerobot.cli import (
    add_ik_cli_args,
    ik_kwargs_from_args,
    parse_rest_pose_env,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teleop_bi_yam import ARM_DOFS, _command_arm, ramp_to_rest  # noqa: E402
from record_bi_yam import (  # noqa: E402
    DEFAULT_REPO_OWNER,
    _action_vector as _joint_action_vector,
    _discard_open_episode,
    _feature_names as _joint_feature_names,
    _open_cameras,
    _push_to_hub,
)
from record_yam_bspline_format import BsplineEpisodeWriter, TcpFk  # noqa: E402


def _decimation(control_freq: float, fps: int, label: str) -> int:
    """Ticks per recorded frame; requires exact division.

    An inexact ratio means the frames land at a rate the dataset does not
    declare — LeRobot computes timestamps as frame_index/fps, and their mp4 is
    written at the nominal fps, so both would drift against wall clock.
    """
    ratio = control_freq / fps
    decim = int(round(ratio))
    if decim < 1 or abs(ratio - decim) > 1e-9:
        divisors = [d for d in range(1, int(control_freq) + 1)
                    if abs(control_freq / d - round(control_freq / d)) < 1e-9
                    and round(control_freq / d) <= 60]
        raise SystemExit(
            f"--{label} {fps} does not divide --control-freq {control_freq:g} exactly. "
            f"Valid rates at this control frequency: "
            f"{sorted({int(control_freq / d) for d in divisors})}")
    return decim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("right", "left"), default="right")
    ap.add_argument("--num-episodes", type=int, default=0,
                    help="end the session after this many saved episodes (0 = until Ctrl-C)")
    ap.add_argument("--task", default="teleop",
                    help="language instruction stored on every LeRobot frame")
    # LeRobot side
    ap.add_argument("--repo-id", default=None,
                    help="LeRobot dataset id. Default: atharva/yam-dual-<stamp>")
    ap.add_argument("--root", default=None, help="local LeRobot dataset root")
    ap.add_argument("--lerobot-fps", type=int, default=25,
                    help="LeRobot sample rate; must divide --control-freq (default: 25)")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete an existing LeRobot dataset dir instead of erroring")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing dataset pair instead of creating one. "
                         "--num-episodes then counts episodes added THIS session. The "
                         "bspline side appends by nature (one dir per episode), so both "
                         "stay in lockstep. Cameras, fps and features must be unchanged.")
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--private", action="store_true")
    # bspline side
    ap.add_argument("--bspline-dir", default="data/demos-ee",
                    help="episode root for the EE-only dataset (default: data/demos-ee)")
    ap.add_argument("--bspline-fps", type=int, default=10,
                    help="bspline sample rate; must divide --control-freq (default: 10)")
    # shared
    ap.add_argument("--control-freq", type=float, default=200.0,
                    help="teleop/IK/command rate (default: 200)")
    ap.add_argument("--left-can", default="can1")
    ap.add_argument("--right-can", default="can0")
    ap.add_argument("--sim", action="store_true", help="i2rt sim robot, no hardware")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws")
    ap.add_argument("--rest-duration-s", type=float, default=3.0)
    ap.add_argument("--rest-steps", type=int, default=90)
    add_ik_cli_args(ap)
    args = ap.parse_args()

    setup_logging(level=logging.INFO)
    logger = get_logger(__name__, "session")

    lerobot_decim = _decimation(args.control_freq, args.lerobot_fps, "lerobot-fps")
    bspline_decim = _decimation(args.control_freq, args.bspline_fps, "bspline-fps")

    session_start = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(session_start))
    if args.repo_id is None:
        args.repo_id = f"{DEFAULT_REPO_OWNER}/yam-dual-{stamp}"
        logger.info("no --repo-id given; using default %s", args.repo_id)

    try:
        from i2rt.robots.get_robot import get_yam_robot
    except ImportError as e:
        raise ImportError(
            "i2rt is required. Clone https://github.com/i2rt-robotics/i2rt "
            "and install it (pip install -e path/to/i2rt)."
        ) from e
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    hand = args.arm
    hands = (hand,)
    channel = args.right_can if hand == "right" else args.left_can

    fk = TcpFk()  # before anything else, so a missing YAM_XML fails fast

    lerobot_root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    if lerobot_root.exists():
        if args.overwrite:
            logger.warning("--overwrite: removing %s", lerobot_root)
            shutil.rmtree(lerobot_root)
        elif not args.resume:
            logger.error("LeRobot dataset already exists at %s — pass --resume to "
                         "append more episodes, --overwrite to start fresh, or "
                         "choose a different --repo-id", lerobot_root)
            sys.exit(1)
    elif args.resume:
        logger.error("--resume given but no dataset at %s — drop the flag to create it",
                     lerobot_root)
        sys.exit(1)

    resuming = args.resume and lerobot_root.exists()

    bspline_dir = Path(args.bspline_dir)
    bspline_dir.mkdir(parents=True, exist_ok=True)
    existing = [d for d in bspline_dir.iterdir() if d.is_dir()]
    if existing and not resuming:
        logger.warning("%s already holds %d episode(s) — this session appends to them, "
                       "so the two datasets will NOT have matching episode counts",
                       bspline_dir, len(existing))

    cameras, cam_features = _open_cameras(not args.no_cameras, hands, logger)

    names = _joint_feature_names(hands)
    if resuming:
        # Append to the existing dataset. The bspline side appends by nature
        # (one timestamped dir per episode), so both continue in lockstep.
        dataset = LeRobotDataset.resume(
            repo_id=args.repo_id,
            root=args.root,
            image_writer_threads=4 * len(cameras),
        )
        if dataset.fps != args.lerobot_fps:
            logger.error("existing dataset is %d fps but --lerobot-fps is %d — "
                         "refusing to mix rates in one dataset",
                         dataset.fps, args.lerobot_fps)
            sys.exit(1)
        if dataset.num_episodes != len(existing):
            logger.warning("resuming with mismatched counts: LeRobot has %d episode(s), "
                           "bspline dir has %d. They will stay offset by %+d.",
                           dataset.num_episodes, len(existing),
                           dataset.num_episodes - len(existing))
        logger.info("RESUMING: %d episode(s) already recorded; this session adds %s more",
                    dataset.num_episodes,
                    args.num_episodes if args.num_episodes else "unlimited")
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.lerobot_fps,
            root=args.root,
            robot_type="yam",
            use_videos=bool(cam_features),
            image_writer_threads=4 * len(cameras),
            features={
                "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
                "action": {"dtype": "float32", "shape": (len(names),), "names": names},
                **cam_features,
            },
        )

    logger.info("──────────────────────────────────────────────────")
    logger.info("%s arm, %.0f Hz control loop", hand, args.control_freq)
    logger.info("  LeRobot  %2d Hz (every %2d ticks)  %d-dim joints -> %s",
                args.lerobot_fps, lerobot_decim, len(names), dataset.root)
    logger.info("  bspline  %2d Hz (every %2d ticks)  EE pose        -> %s",
                args.bspline_fps, bspline_decim, bspline_dir.resolve())
    logger.info("  cameras  %s (uncropped, native resolution, in BOTH datasets)",
                list(cameras) or "none")
    logger.info("  right B start / discard · left Y save · end after %s",
                f"{args.num_episodes} episode(s)" if args.num_episodes else "Ctrl-C")
    logger.info("──────────────────────────────────────────────────")

    rest = parse_rest_pose_env(
        "RIGHT_REST_POSE" if hand == "right" else "LEFT_REST_POSE", [0.0] * ARM_DOFS)
    robot = get_yam_robot(channel=channel, sim=args.sim)

    teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(
        id="vr-record-dual",
        ws_url=args.ws_url,
        rest_qpos_left=rest if hand == "left" else [0.0] * ARM_DOFS,
        rest_qpos_right=rest if hand == "right" else [0.0] * ARM_DOFS,
        **ik_kwargs_from_args(args),
    ))
    teleop.connect()
    arm_log = get_logger(f"{__name__}.{hand}", f"arm{hand}")
    ramp_to_rest(robot, rest, args.rest_duration_s, args.rest_steps, arm_log, hand)

    recording = False
    writer: BsplineEpisodeWriter | None = None
    pending: list[BsplineEpisodeWriter] = []
    lerobot_frames = bspline_frames = 0
    saved_this_session = 0
    episode_start = 0.0
    last_b = last_y = False
    last_rgb: dict[str, np.ndarray] = {}
    tick = 0
    period = 1.0 / args.control_freq
    interrupted = False

    def rgb(cam_id: str) -> np.ndarray | None:
        """Latest frame as RGB, reusing the previous one on a dropped grab.
        cv2 gives BGR; both writers store RGB."""
        img = cameras[cam_id].latest() if cam_id in cameras else None
        if img is None:
            return last_rgb.get(cam_id)
        out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        last_rgb[cam_id] = out
        return out

    try:
        next_tick = time.perf_counter()
        while True:
            b = teleop.is_pause_pressed()     # right B
            y = teleop.is_reverse_pressed()   # left Y

            if b and not last_b:
                if recording:
                    logger.info("✗ episode DISCARDED (%d LeRobot / %d bspline frames, "
                                "%.1fs) — restarting now", lerobot_frames, bspline_frames,
                                time.perf_counter() - episode_start)
                    _discard_open_episode(dataset, logger)
                    writer = None  # its dir is only created on flush, so this is enough
                recording = True
                lerobot_frames = bspline_frames = 0
                tick = 0
                writer = BsplineEpisodeWriter(bspline_dir, args.bspline_fps)
                episode_start = time.perf_counter()
                logger.info("● episode %d STARTED at %s", dataset.num_episodes,
                            time.strftime("%H:%M:%S"))

            if y and not last_y and recording:
                if lerobot_frames == 0 or bspline_frames == 0:
                    logger.warning("Y pressed with %d/%d frames — still recording",
                                   lerobot_frames, bspline_frames)
                else:
                    dur = time.perf_counter() - episode_start
                    dataset.save_episode()
                    assert writer is not None
                    writer.flush_async()
                    pending.append(writer)
                    saved_bspline = len(pending) + len(existing)
                    logger.info("✓ episode %d SAVED (%.1fs) — LeRobot %d frames @ %.1f Hz, "
                                "bspline %d frames @ %.1f Hz -> %s",
                                dataset.num_episodes - 1, dur,
                                lerobot_frames, lerobot_frames / dur if dur else 0.0,
                                bspline_frames, bspline_frames / dur if dur else 0.0,
                                writer.episode_dir.name)
                    if dataset.num_episodes != saved_bspline:
                        logger.warning("episode counts diverged: LeRobot %d, bspline %d "
                                       "(pre-existing episodes in %s?)",
                                       dataset.num_episodes, saved_bspline, bspline_dir)
                    writer = None
                    recording = False
                    saved_this_session += 1
                    # Counts episodes saved in THIS session, not the dataset
                    # total — otherwise resuming at 100 with --num-episodes 100
                    # would end before recording anything.
                    if args.num_episodes and saved_this_session >= args.num_episodes:
                        logger.info("reached --num-episodes=%d for this session "
                                    "(%d in the dataset) — ending",
                                    args.num_episodes, dataset.num_episodes)
                        break
            last_b, last_y = b, y

            # Every control tick: observe, act, command. One get_observations()
            # per tick, shared by the haptics and both writers.
            robot_obs = robot.get_observations()
            action = teleop.get_action()
            _command_arm(robot, action, f"{hand}_")
            try:
                torques = {}
                if "gripper_eff" in robot_obs:
                    torques[f"{hand}_gripper.torque"] = float(robot_obs["gripper_eff"][0])
                if "gripper_pos" in robot_obs:
                    torques[f"{hand}_gripper.pos"] = 1.0 - float(robot_obs["gripper_pos"][0])
                if torques:
                    teleop.send_feedback({"torques": torques})
            except Exception:
                pass

            tick += 1
            if recording:
                joints = np.asarray(robot_obs["joint_pos"][:ARM_DOFS], dtype=np.float64)
                gripper = 1.0 - float(robot_obs["gripper_pos"][0])

                if tick % lerobot_decim == 0:
                    state = np.asarray([*joints, gripper], dtype=np.float32)
                    frame = {
                        "observation.state": state,
                        "action": _joint_action_vector(action, hands),
                        "task": args.task,
                    }
                    for cam_id in cameras:
                        img = rgb(cam_id)
                        if img is not None:
                            frame[f"observation.images.{cam_id}"] = img
                    dataset.add_frame(frame)
                    lerobot_frames += 1

                if tick % bspline_decim == 0 and writer is not None:
                    arm_pos, arm_quat = fk.pose(joints)
                    cmd_joints = [float(action[f"{hand}_joint_{j + 1}.pos"])
                                  for j in range(ARM_DOFS)]
                    cmd_pos, cmd_quat = fk.pose(cmd_joints)
                    obs_dict = {
                        "arm_pos": arm_pos,
                        "arm_quat": arm_quat,
                        "gripper_pos": np.array([gripper], dtype=np.float64),
                    }
                    for cam_id in cameras:
                        img = rgb(cam_id)
                        if img is not None:
                            key = "wrist_image" if cam_id.endswith("_wrist") else f"{cam_id}_image"
                            obs_dict[key] = img
                    writer.step(obs_dict, {
                        "arm_pos": cmd_pos,
                        "arm_quat": cmd_quat,
                        "gripper_pos": np.array(
                            [float(action[f"{hand}_gripper.pos"])], dtype=np.float64),
                    })
                    bspline_frames += 1

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        interrupted = True
        logger.info("interrupted — parking arm at home pose (Ctrl-C again to power off)")
    finally:
        if recording:
            _discard_open_episode(dataset, logger)
            logger.info("open episode discarded from BOTH datasets (ended mid-episode)")
        for w in pending:
            w.wait_for_flush()
        if hasattr(dataset, "finalize"):
            dataset.finalize()
        elapsed = time.time() - session_start
        logger.info("session done: +%d episode(s) this session, %d total, %s elapsed",
                    saved_this_session, dataset.num_episodes,
                    time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        logger.info("  LeRobot %s (v%s) -> %s", args.repo_id, CODEBASE_VERSION, dataset.root)
        logger.info("  bspline -> %s", bspline_dir.resolve())

        try:
            ramp_to_rest(robot, rest, args.rest_duration_s, args.rest_steps, arm_log, hand)
            if interrupted:
                logger.info("arm parked at home — press Ctrl-C again to power off")
                while True:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("powering off")

        try:
            teleop.disconnect()
        finally:
            for r in cameras.values():
                r.stop()
            robot.close()

        if args.push_to_hub:
            _push_to_hub(dataset, args.repo_id, args.private, logger)


if __name__ == "__main__":
    main()
