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
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from vr_teleop_kit.log import get_logger, setup_logging
from vr_teleop_kit.relay.capture import CameraReader, build_camera_specs
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

# Owner prefix used when --repo-id is omitted. The full default id gets a
# wall-clock stamp appended (yam-teleop-<YYYYmmdd-HHMMSS>) so back-to-back
# sessions land in fresh folders — LeRobotDataset.create errors on an
# existing repo.
DEFAULT_REPO_OWNER = "atharva"


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


# Seconds to wait for a camera's first frame before giving up on it.
CAMERA_WARMUP_S = 5.0


def _open_cameras(enabled: bool, logger) -> tuple[dict[str, CameraReader], dict[str, dict]]:
    """Open every discovered camera (top / left_wrist / right_wrist).

    Returns ``(readers, features)`` where ``readers`` maps camera id →
    CameraReader and ``features`` maps the LeRobot feature key
    (``observation.images.<id>``) → its feature dict. Both empty when
    disabled or when no camera produces a frame.

    NB: a v4l2 device can only be opened by one process, and the relay
    lazily grabs the same cameras on the first WebRTC request. So while
    recording with cameras, do NOT enable the camera stream in the Quest
    UI, or open() here will fail with "cannot open camera".

    The feature shape is read from the first actual frame so it matches
    whatever the capture thread emits after any configured rotation.
    """
    if not enabled:
        return {}, {}
    readers: dict[str, CameraReader] = {}
    features: dict[str, dict] = {}
    for spec in build_camera_specs():
        reader = CameraReader(spec)
        frame = None
        deadline = time.time() + CAMERA_WARMUP_S
        while time.time() < deadline:
            frame = reader.latest()
            if frame is not None:
                break
            time.sleep(0.05)
        if frame is None:
            logger.warning("camera %s produced no frame in %.0fs — skipping",
                           spec.id, CAMERA_WARMUP_S)
            reader.stop()
            continue
        h, w = frame.shape[:2]
        readers[spec.id] = reader
        features[f"observation.images.{spec.id}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
        logger.info("camera %s recording at %dx%d (%s)", spec.id, w, h, spec.label)
    return readers, features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=None,
                    help="dataset repo id (e.g. you/yam-towels). "
                         "Default: atharva/yam-teleop-<YYYYmmdd-HHMMSS>")
    ap.add_argument("--task", default="teleop",
                    help="task string stored with every frame (default: 'teleop')")
    ap.add_argument("--root", default=None, help="local dataset root (default: LeRobot's)")
    ap.add_argument("--overwrite", action="store_true",
                    help="if the dataset dir already exists, delete it and start fresh "
                         "(handy for repeated camera checks that reuse a --repo-id)")
    ap.add_argument("--fps", type=int, default=30, help="dataset + control-loop rate")
    ap.add_argument("--num-episodes", type=int, default=0,
                    help="stop after this many saved episodes (0 = until ended from VR)")
    ap.add_argument("--left-can", default="can1", help="left arm CAN interface (default: can1)")
    ap.add_argument("--right-can", default="can0", help="right arm CAN interface (default: can0)")
    ap.add_argument("--arm", choices=("both", "left", "right"), default="both")
    ap.add_argument("--sim", action="store_true",
                    help="use i2rt sim robots (rehearse the recording flow, no hardware)")
    ap.add_argument("--no-cameras", action="store_true",
                    help="skip camera capture (record state/action only). By "
                         "default every discovered camera is recorded as video. "
                         "Do not enable the Quest camera stream while recording "
                         "with cameras — the relay and recorder can't share a v4l2 device.")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws", help="relay server WS URL")
    ap.add_argument("--rest-duration-s", type=float, default=3.0)
    ap.add_argument("--rest-steps", type=int, default=90)
    add_ik_cli_args(ap)
    args = ap.parse_args()

    setup_logging(level=logging.INFO)
    logger = get_logger(__name__, "session")

    # Wall-clock session stamp: names the default repo and anchors the logs.
    session_start = time.time()
    session_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(session_start))
    if args.repo_id is None:
        args.repo_id = f"{DEFAULT_REPO_OWNER}/yam-teleop-{session_stamp}"
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

    # LeRobot's create() raises a bare FileExistsError if the dir exists; catch
    # it here so a reused --repo-id gives an actionable message (or is cleared
    # with --overwrite) instead of a stack trace.
    dataset_root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    if dataset_root.exists():
        if args.overwrite:
            logger.warning("--overwrite: removing existing dataset at %s", dataset_root)
            shutil.rmtree(dataset_root)
        else:
            logger.error(
                "dataset already exists at %s — pass --overwrite to replace it, "
                "or choose a different --repo-id", dataset_root)
            sys.exit(1)

    hands: tuple[str, ...] = ("left", "right") if args.arm == "both" else (args.arm,)
    channels = {"left": args.left_can, "right": args.right_can}

    cameras, cam_features = _open_cameras(not args.no_cameras, logger)
    use_videos = bool(cam_features)

    names = _feature_names(hands)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,
        robot_type="bi_yam" if args.arm == "both" else "yam",
        use_videos=use_videos,
        # Encode camera frames off the control thread so add_frame() doesn't
        # stall the fps loop; no-op when there are no cameras.
        image_writer_threads=4 * len(cameras),
        features={
            "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
            "action": {"dtype": "float32", "shape": (len(names),), "names": names},
            **cam_features,
        },
    )
    logger.info("dataset %s — LeRobot codebase %s, %d fps, %d-dim state/action, cameras=%s, videos=%s",
                args.repo_id, CODEBASE_VERSION, args.fps, len(names),
                list(cameras) or "none", use_videos)
    logger.info("session started %s — writing to %s",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_start)), dataset.root)

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
    arm_logs = {h: get_logger(f"{__name__}.{h}", f"arm{h}") for h in hands}
    for h in hands:
        ramp_to_rest(robots[h], rests[h], args.rest_duration_s, args.rest_steps, arm_logs[h], h)

    logger.info("──────────────────────────────────────────────────")
    logger.info("VR recording controls:")
    logger.info("  right B        start / save episode")
    logger.info("  left Y         discard current episode")
    logger.info("  B + Y (hold)   end session (while idle)")
    logger.info("──────────────────────────────────────────────────")

    recording = False
    frames_in_episode = 0
    episode_start = 0.0
    # Last good frame per camera, so a momentary dropped grab reuses the
    # previous frame rather than leaving a feature missing from add_frame().
    last_frames: dict[str, np.ndarray] = {cid: r.latest() for cid, r in cameras.items()}
    total_frames = 0
    last_b = last_y = False
    both_held_since: float | None = None
    period = 1.0 / args.fps
    interrupted = False

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
                        episode_start = time.perf_counter()
                        logger.info("● episode %d STARTED at %s",
                                    dataset.num_episodes, time.strftime("%H:%M:%S"))
                    elif frames_in_episode > 0:
                        dur = time.perf_counter() - episode_start
                        total_frames += frames_in_episode
                        dataset.save_episode()
                        recording = False
                        logger.info("✓ episode %d SAVED (%d frames, %.1fs, %.1f fps) — %d frames total",
                                    dataset.num_episodes - 1, frames_in_episode, dur,
                                    frames_in_episode / dur if dur > 0 else 0.0, total_frames)
                        if args.num_episodes and dataset.num_episodes >= args.num_episodes:
                            logger.info("reached --num-episodes=%d", args.num_episodes)
                            break
                if y and not last_y and recording:
                    dur = time.perf_counter() - episode_start
                    dataset.clear_episode_buffer()
                    recording = False
                    logger.info("✗ episode DISCARDED (%d frames, %.1fs) — go again",
                                frames_in_episode, dur)
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
                frame = {
                    "observation.state": obs,
                    "action": _action_vector(action, hands),
                    "task": args.task,
                }
                for cid, reader in cameras.items():
                    img = reader.latest()
                    if img is None:
                        img = last_frames.get(cid)
                    if img is None:
                        continue
                    last_frames[cid] = img
                    # cv2 grabs BGR; LeRobot's image writer treats a 3-channel
                    # frame as RGB, so convert or red/blue come out swapped.
                    frame[f"observation.images.{cid}"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dataset.add_frame(frame)
                frames_in_episode += 1

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        interrupted = True
        logger.info("interrupted — parking arms at home pose (Ctrl-C again to power off)")
    finally:
        if recording:
            dataset.clear_episode_buffer()
            logger.info("open episode discarded (session ended mid-episode)")
        if hasattr(dataset, "finalize"):
            dataset.finalize()
        elapsed = time.time() - session_start
        logger.info("session done: %d episode(s), %d frames, %s elapsed",
                    dataset.num_episodes, total_frames,
                    time.strftime("%H:%M:%S", time.gmtime(elapsed)))
        logger.info("dataset %s (LeRobot %s) at %s",
                    args.repo_id, CODEBASE_VERSION, dataset.root)

        # First Ctrl-C brings us here with the arms still powered: ramp them
        # back to the home/rest pose and hold there. A second Ctrl-C (or a
        # session ended cleanly from VR) then powers the arms off. A Ctrl-C
        # during the ramp itself skips straight to power-off.
        try:
            for h in hands:
                ramp_to_rest(robots[h], rests[h], args.rest_duration_s,
                             args.rest_steps, arm_logs[h], h)
            if interrupted:
                logger.info("arms parked at home — press Ctrl-C again to power off")
                while True:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("powering off")

        try:
            teleop.disconnect()
        finally:
            for r in cameras.values():
                r.stop()
            for h in hands:
                robots[h].close()


if __name__ == "__main__":
    main()
