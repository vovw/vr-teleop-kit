"""Record LeRobot datasets from bimanual YAM teleop — controlled entirely
from the Quest controllers, no keyboard needed.

In-VR control map (on top of the normal teleop bindings):

    grip (squeeze)     drive the arm (clutch), trigger = gripper
    A / X (hold)       precision scale
    thumbstick click   ramp that arm to its rest pose (nice between episodes)
    right B (idle)     start an episode
    right B (recording) throw the take away and restart it immediately
    left Y             save the episode being recorded

The session ends by itself after `--num-episodes` saved episodes (or on
Ctrl-C, which parks the arms first). There is no button chord.

Teleop keeps driving the robot between episodes (so you can reset the
scene with the arm), but frames are only written while an episode is
open. Episode transitions are logged loudly; watch the terminal or trust
the muscle memory: B ... do the thing ... Y.

The loop runs at the dataset fps (default 30), like `lerobot-record`.
The per-joint velocity caps are per-tick, so at 30 fps the arm is slower
than in the 200 Hz examples — raise the caps from the web Settings panel
if the arm feels sluggish while recording.

Requires lerobot (this is the one entry point that genuinely needs it —
on a weak connection install CPU-only torch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

Frames land in LeRobot v3.0 layout: `data/` as parquet, cameras as mp4.
With `--push-to-hub` the finished dataset is uploaded to the Hugging Face
Hub once the arms are powered down — public by default, `--private` to
opt out. The `--repo-id` owner must be the account you are logged into
(`hf auth login`), or the upload 403s.

Run (hardware):
    python examples/record_bi_yam.py \\
        --repo-id you/yam-task --task "fold the towel" \\
        --left-can can_left --right-can can_right \\
        --num-episodes 20 --push-to-hub

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


def _discard_open_episode(dataset, logger) -> None:
    """Drop the in-progress episode buffer *and* its temp PNG frames.

    LeRobot's ``clear_episode_buffer(delete_images=True)`` only deletes dirs
    for features with ``dtype == "image"`` (``meta.image_keys``). Our cameras
    are recorded as ``dtype == "video"``, so that set is empty and the raw
    per-frame PNGs the async image writer dumped to ``images/`` are left on
    disk — ~2 MB/frame/camera, i.e. gigabytes for one discarded take.

    So we also invoke ``cleanup_interrupted_episode`` (which iterates
    ``meta.camera_keys`` and covers video features) for the still-open episode
    index. Both calls are guarded and idempotent, so this stays correct if a
    future LeRobot fixes the ``clear_episode_buffer`` behavior.
    """
    # The open (unsaved) episode takes the next index: saved episodes occupy
    # 0..num_episodes-1, so the buffer being cleared is num_episodes.
    ep_index = dataset.num_episodes
    dataset.clear_episode_buffer()
    writer = getattr(dataset, "writer", None)
    cleanup = getattr(writer, "cleanup_interrupted_episode", None)
    if cleanup is None:
        return
    try:
        cleanup(ep_index)
    except Exception:
        logger.debug("cleanup_interrupted_episode(%d) failed", ep_index, exc_info=True)


def _push_to_hub(dataset, repo_id: str, private: bool, logger) -> None:
    """Upload the finished dataset to the Hugging Face Hub.

    Best-effort by design: the dataset is already complete on disk before
    this runs, so a failed upload must not mask the session result. It is
    logged and the local copy left untouched for a manual retry.

    Called after the arms are powered down — a few-hundred-MB video upload
    takes minutes, and there is no reason to hold the motors energized for
    it.
    """
    if dataset.num_episodes == 0:
        logger.info("no episodes saved — nothing to push")
        return

    # A repo id whose owner isn't the logged-in user 403s after uploading
    # nothing useful; warn up front rather than after the wait. This is easy
    # to hit because an omitted --repo-id defaults to DEFAULT_REPO_OWNER.
    owner = repo_id.split("/")[0] if "/" in repo_id else None
    try:
        from huggingface_hub import whoami

        user = whoami().get("name")
    except Exception:  # not logged in, offline, or hub API changed
        user = None
    if user and owner and owner != user:
        logger.warning(
            "--repo-id owner %r is not the logged-in HF user %r — this push will "
            "likely fail with 403. Use --repo-id %s/<name> next time.",
            owner, user, user,
        )

    logger.info("pushing %d episode(s) to https://huggingface.co/datasets/%s (%s)",
                dataset.num_episodes, repo_id, "private" if private else "public")
    try:
        dataset.push_to_hub(private=private, tags=["robotics", "lerobot", "yam", "teleop"])
    except Exception:
        logger.exception("push failed — dataset kept at %s (retry by hand)", dataset.root)
    else:
        logger.info("push complete: https://huggingface.co/datasets/%s", repo_id)


def _open_cameras(
    enabled: bool, hands: tuple[str, ...], logger
) -> tuple[dict[str, CameraReader], dict[str, dict]]:
    """Open the discovered cameras relevant to ``hands``.

    The overhead ``top`` camera is always kept; a wrist camera is kept only
    when its arm is actually being recorded. Without this filter a
    ``--arm right`` run still records ``left_wrist`` — a second video stream
    of an arm that never moves, roughly doubling the dataset size for
    nothing.

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
    wanted = {"top", *(f"{hand}_wrist" for hand in hands)}
    readers: dict[str, CameraReader] = {}
    features: dict[str, dict] = {}
    for spec in build_camera_specs():
        if spec.id not in wanted:
            logger.info("camera %s skipped — %s arm not in --arm %s",
                        spec.id, spec.id.removesuffix("_wrist"), "/".join(hands))
            continue
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
                    help="end the session after this many saved episodes "
                         "(0 = keep going until Ctrl-C)")
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
    ap.add_argument("--push-to-hub", action="store_true",
                    help="upload the dataset to the Hugging Face Hub when the "
                         "session ends (after the arms are powered down). "
                         "Public unless --private is also passed.")
    ap.add_argument("--private", action="store_true",
                    help="with --push-to-hub, create the Hub repo private "
                         "(default: public)")
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

    cameras, cam_features = _open_cameras(not args.no_cameras, hands, logger)
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
    logger.info("  right B        start episode (idle) / discard + restart it (recording)")
    logger.info("  left Y         save episode")
    logger.info("  end of session %s",
                f"after {args.num_episodes} saved episode(s)"
                if args.num_episodes else "on Ctrl-C (no --num-episodes given)")
    logger.info("──────────────────────────────────────────────────")

    recording = False
    frames_in_episode = 0
    episode_start = 0.0
    # Last good frame per camera, so a momentary dropped grab reuses the
    # previous frame rather than leaving a feature missing from add_frame().
    last_frames: dict[str, np.ndarray] = {cid: r.latest() for cid, r in cameras.items()}
    total_frames = 0
    last_b = last_y = False
    period = 1.0 / args.fps
    interrupted = False

    try:
        next_tick = time.perf_counter()
        while True:
            b = teleop.is_pause_pressed()     # right B (level)
            y = teleop.is_reverse_pressed()   # left Y (level)

            # Right B: start when idle, discard-and-restart when recording. The
            # restart is immediate (recording stays True) so a botched take is
            # re-done with one press instead of stop-then-start.
            if b and not last_b:
                if recording:
                    dur = time.perf_counter() - episode_start
                    _discard_open_episode(dataset, logger)
                    logger.info("✗ episode DISCARDED (%d frames, %.1fs) — restarting now",
                                frames_in_episode, dur)
                recording, frames_in_episode = True, 0
                episode_start = time.perf_counter()
                logger.info("● episode %d STARTED at %s",
                            dataset.num_episodes, time.strftime("%H:%M:%S"))

            # Left Y: save. An empty buffer can't be saved (LeRobot raises), so
            # a Y before any frame landed is ignored and the episode stays open.
            if y and not last_y and recording:
                if frames_in_episode == 0:
                    logger.warning("Y pressed with 0 frames recorded — still recording")
                else:
                    dur = time.perf_counter() - episode_start
                    total_frames += frames_in_episode
                    dataset.save_episode()
                    recording = False
                    logger.info("✓ episode %d SAVED (%d frames, %.1fs, %.1f fps) — %d frames total",
                                dataset.num_episodes - 1, frames_in_episode, dur,
                                frames_in_episode / dur if dur > 0 else 0.0, total_frames)
                    if args.num_episodes and dataset.num_episodes >= args.num_episodes:
                        logger.info("reached --num-episodes=%d — ending session",
                                    args.num_episodes)
                        break
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
            _discard_open_episode(dataset, logger)
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

        # Last, with the motors off: the upload can take minutes.
        if args.push_to_hub:
            _push_to_hub(dataset, args.repo_id, args.private, logger)


if __name__ == "__main__":
    main()
