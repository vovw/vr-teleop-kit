"""Record single-arm EE-pose demos in the bspline/tidybot2 `real_env` format.

Drives ONE YAM from the Quest (the right controller drives the right arm by
default) and writes episodes in the exact on-disk layout that
`bspline-policy/real_env/yam_teleop/episode_storage.py` produces, so the data
feeds straight into that stack:

    <output-dir>/<YYYYmmddTHHMMSS%f>/
        wrist_image.mp4      H.264, one frame per step, at --freq fps
        data.pkl             {'timestamps': [...], 'observations': [...],
                              'actions': [...]}   (images nulled, see below)

Per step, matching their single-YAM `RealEnv.get_obs()` / `TeleopController`:

    observation  {'arm_pos': (3,) float64,          TCP position, arm base frame
                  'arm_quat': (4,) float64,         xyzw, w >= 0
                  'gripper_pos': (1,) float64,      0 = open, 1 = closed
                  'wrist_image': (480, 640, 3) uint8 RGB}
    action       {'arm_pos', 'arm_quat', 'gripper_pos'}   same shapes, no image

`data.pkl` stores each image entry as `None` (the frames live in the mp4), which
is what their `EpisodeReader` expects when it re-attaches them. That means
`reviewer.py`, `sort_demos_from_review.py` and `convert_to_robomimic_hdf5.py`
all work on this output unchanged, and the resulting HDF5 lands on their
`single_yam_rot6d` schema: obs `{arm_pos, arm_quat, gripper_pos, wrist_image}`
with a 10-dim action (pos 3 + rot6d 6 + gripper 1).

In-VR controls (same as the other recorders):

    grip (squeeze)      drive the arm (clutch), trigger = gripper
    thumbstick click    ramp the arm to its rest pose (between episodes)
    right B             start an episode / discard + restart while recording
    left Y              save the episode

Note the asymmetry: start is on the RIGHT controller, save on the LEFT, so keep
both in hand even though only one arm moves.

TCP convention
--------------
`arm_pos`/`arm_quat` are the link6 flange re-axed by their fixed
`_T_LINK6_TO_TCP` (a 90 degree rotation about z, zero translation) — the same
point and axes their `yam_server.get_state()` reports, NOT the fingertip
`tool0` site the other examples here use.

The pose is computed from this repo's MuJoCo model, not their pyroki/URDF
chain. The two models disagree by up to ~9 mm on the same joint angles, so
these numbers are not interchangeable with episodes recorded by their
`yam_server`: do not mix the two datasets, and do not deploy a policy trained
on this data through their stack without resolving that first. Collected and
deployed entirely within this repo the data is self-consistent, which is all a
policy needs — it never sees an IK solver.

Run (hardware, right arm):
    python examples/record_yam_bspline_format.py \\
        --arm right --right-can can0 --num-episodes 1

Rehearse with no motors and no camera (black wrist frames, exactly as their
RealEnv does when cameras are disabled):
    python examples/record_yam_bspline_format.py --sim --no-cameras --num-episodes 1
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import numpy as np

from vr_teleop_kit.ik.model import build_model_with_tool0_site
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teleop_bi_yam import ARM_DOFS, _command_arm, _gripper_feedback, ramp_to_rest  # noqa: E402

# Their raw capture resolution (constants.RAW_IMAGE_WIDTH/HEIGHT). Frames are
# resized to this so the mp4 matches what their pipeline expects downstream.
RAW_IMAGE_WIDTH, RAW_IMAGE_HEIGHT = 640, 480

# Fixed link6 -> TCP transform, copied from their `yam_server.py`. Pure
# rotation: their TCP sits at the link6 flange origin, re-axed.
_T_LINK6_TO_TCP = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


class TcpFk:
    """Forward kinematics to their TCP frame, off this repo's MuJoCo model.

    A dedicated model instance: the teleop's solvers share their MuJoCo `data`
    between fk(), solve() and j4_anchor_xpos(), so borrowing one of those
    mid-tick would perturb the clutch anchor.
    """

    def __init__(self, model_path=None) -> None:
        self.model, self.data = build_model_with_tool0_site(model_path)
        self.link6 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        if self.link6 == -1:
            raise RuntimeError("link6 body missing from the YAM model")

    def pose(self, joints) -> tuple[np.ndarray, np.ndarray]:
        """(pos (3,), quat xyzw (4,) with w >= 0) of the TCP, arm base frame."""
        self.data.qpos[:] = 0.0
        self.data.qpos[:ARM_DOFS] = np.asarray(joints, dtype=np.float64)[:ARM_DOFS]
        mujoco.mj_kinematics(self.model, self.data)

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = self.data.xpos[self.link6]
        T[:3, :3] = self.data.xmat[self.link6].reshape(3, 3)
        T = T @ _T_LINK6_TO_TCP

        quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(quat_wxyz, np.ascontiguousarray(T[:3, :3]).reshape(9))
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        # Their TeleopController and yam_server both enforce this hemisphere.
        if quat_xyzw[3] < 0.0:
            quat_xyzw = -quat_xyzw
        return T[:3, 3].copy(), quat_xyzw


def _write_frames_to_mp4(frames: list[np.ndarray], mp4_path: Path, fps: int) -> None:
    """H.264 via system ffmpeg, RGB in — same encode as their
    `episode_storage.write_frames_to_mp4`, with the same OpenCV fallback."""
    ffmpeg = shutil.which("ffmpeg")
    height, width, _ = frames[0].shape
    if ffmpeg:
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(mp4_path)]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read()
            if proc.wait() == 0:
                return
            if stderr:
                print(stderr.decode(errors="replace"), end="")
        except (OSError, BrokenPipeError):
            pass
    out = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"avc1"),
                          fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


class BsplineEpisodeWriter:
    """Writes one episode in their `EpisodeWriter` layout.

    Kept byte-compatible with `bspline-policy/real_env/yam_teleop/episode_storage.py`
    rather than importing it, so this repo stays standalone: same directory
    stamp, same per-key mp4s, same pickle payload with image entries replaced
    by None. Verified by loading the output with their `EpisodeReader`.
    """

    def __init__(self, output_dir: Path, fps: int) -> None:
        self.output_dir = Path(output_dir)
        self.episode_dir = self.output_dir / datetime.now().strftime("%Y%m%dT%H%M%S%f")
        assert not self.episode_dir.exists()
        self.fps = fps
        self.timestamps: list[float] = []
        self.observations: list[dict] = []
        self.actions: list[dict] = []
        self.flush_thread: threading.Thread | None = None

    def step(self, obs: dict, action: dict) -> None:
        self.timestamps.append(time.time())
        self.observations.append(obs)
        self.actions.append(action)

    def __len__(self) -> int:
        return len(self.observations)

    def _flush(self) -> None:
        assert len(self) > 0
        self.episode_dir.mkdir(parents=True)

        # Pull image observations out into per-key mp4s, leaving None behind.
        frames_dict: dict[str, list[np.ndarray]] = {}
        for obs in self.observations:
            for k, v in obs.items():
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    frames_dict.setdefault(k, []).append(v)
                    obs[k] = None
        for k, frames in frames_dict.items():
            _write_frames_to_mp4(frames, self.episode_dir / f"{k}.mp4", self.fps)

        with open(self.episode_dir / "data.pkl", "wb") as f:
            pickle.dump({"timestamps": self.timestamps,
                         "observations": self.observations,
                         "actions": self.actions}, f)

    def flush_async(self) -> None:
        self.flush_thread = threading.Thread(target=self._flush, daemon=True)
        self.flush_thread.start()

    def wait_for_flush(self) -> None:
        if self.flush_thread is not None:
            self.flush_thread.join()
            self.flush_thread = None


def _open_wrist_camera(enabled: bool, hand: str, logger) -> CameraReader | None:
    """The wrist camera for the arm in use, recorded as `wrist_image`."""
    if not enabled:
        logger.info("cameras disabled — wrist_image will be black frames "
                    "(same as their RealEnv with use_cameras=False)")
        return None
    want = f"{hand}_wrist"
    for spec in build_camera_specs():
        if spec.id != want:
            continue
        reader = CameraReader(spec)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if reader.latest() is not None:
                logger.info("wrist camera %s (%s) -> wrist_image at %dx%d",
                            spec.id, spec.label, RAW_IMAGE_WIDTH, RAW_IMAGE_HEIGHT)
                return reader
            time.sleep(0.05)
        logger.warning("wrist camera %s produced no frame in 5s — using black frames", want)
        reader.stop()
        return None
    logger.warning("no %s camera discovered — using black frames", want)
    return None


def _wrist_frame(reader: CameraReader | None, last: np.ndarray | None) -> np.ndarray:
    """RGB uint8 at their raw resolution. Falls back to the previous frame on a
    dropped grab, then to black — their RealEnv also returns zeros with no camera."""
    img = reader.latest() if reader is not None else None
    if img is None:
        if last is not None:
            return last
        return np.zeros((RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH, 3), dtype=np.uint8)
    if (img.shape[1], img.shape[0]) != (RAW_IMAGE_WIDTH, RAW_IMAGE_HEIGHT):
        img = cv2.resize(img, (RAW_IMAGE_WIDTH, RAW_IMAGE_HEIGHT))
    # CameraReader hands back BGR; their pipeline stores RGB.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("right", "left"), default="right",
                    help="which arm to drive and record (default: right)")
    ap.add_argument("--output-dir", default="data/demos",
                    help="episode root, their default (default: data/demos)")
    ap.add_argument("--freq", type=int, default=10,
                    help="dataset sample rate; their POLICY_CONTROL_FREQ is 10")
    ap.add_argument("--control-freq", type=float, default=200.0,
                    help="teleop/IK/command rate (default: 200, same as the plain "
                         "teleop example). Kept separate from --freq: commanding the "
                         "arm at the dataset rate makes it jittery, because the arm "
                         "gets a new joint target only every 1/freq s and the IK's "
                         "per-tick velocity cap becomes that much tighter.")
    ap.add_argument("--num-episodes", type=int, default=0,
                    help="end the session after this many saved episodes (0 = until Ctrl-C)")
    ap.add_argument("--left-can", default="can1", help="left arm CAN interface")
    ap.add_argument("--right-can", default="can0", help="right arm CAN interface")
    ap.add_argument("--sim", action="store_true",
                    help="i2rt sim robot — rehearse the flow with no hardware")
    ap.add_argument("--no-cameras", action="store_true",
                    help="skip the wrist camera; wrist_image becomes black frames")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws", help="relay server WS URL")
    ap.add_argument("--rest-duration-s", type=float, default=3.0)
    ap.add_argument("--rest-steps", type=int, default=90)
    add_ik_cli_args(ap)
    args = ap.parse_args()

    setup_logging(level=logging.INFO)
    logger = get_logger(__name__, "session")

    try:
        from i2rt.robots.get_robot import get_yam_robot
    except ImportError as e:
        raise ImportError(
            "i2rt is required. Clone https://github.com/i2rt-robotics/i2rt "
            "and install it (pip install -e path/to/i2rt)."
        ) from e

    hand = args.arm
    channel = args.right_can if hand == "right" else args.left_can
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fk = TcpFk()  # before the robot comes up, so a missing YAM_XML fails fast
    camera = _open_wrist_camera(not args.no_cameras, hand, logger)

    session_start = time.time()
    logger.info("recording %s arm: %.0f Hz control loop, %d Hz dataset "
                "(1 frame every %d ticks) into %s (bspline/tidybot2 format)",
                hand, args.control_freq, args.freq,
                max(1, int(round(args.control_freq / args.freq))), output_dir.resolve())
    logger.info("obs/action = arm_pos(3) + arm_quat(4, xyzw) + gripper_pos(1), "
                "gripper 0=open 1=closed, TCP = link6 flange re-axed")
    logger.warning("poses come from this repo's MuJoCo model, which disagrees with "
                   "their pyroki/URDF chain by up to ~9 mm — self-consistent for "
                   "training here, not mixable with episodes from their yam_server")

    rest = parse_rest_pose_env(
        "RIGHT_REST_POSE" if hand == "right" else "LEFT_REST_POSE",
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    robot = get_yam_robot(channel=channel, sim=args.sim)

    teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(
        id="vr-record-bspline",
        ws_url=args.ws_url,
        rest_qpos_left=rest if hand == "left" else [0.0] * ARM_DOFS,
        rest_qpos_right=rest if hand == "right" else [0.0] * ARM_DOFS,
        **ik_kwargs_from_args(args),
    ))
    teleop.connect()
    arm_log = get_logger(f"{__name__}.{hand}", f"arm{hand}")
    ramp_to_rest(robot, rest, args.rest_duration_s, args.rest_steps, arm_log, hand)

    logger.info("──────────────────────────────────────────────────")
    logger.info("VR recording controls:")
    logger.info("  right B        start episode (idle) / discard + restart it (recording)")
    logger.info("  left Y         save episode")
    logger.info("  end of session %s",
                f"after {args.num_episodes} saved episode(s)"
                if args.num_episodes else "on Ctrl-C (no --num-episodes given)")
    logger.info("──────────────────────────────────────────────────")

    writer: BsplineEpisodeWriter | None = None
    saved_episodes = 0
    total_frames = 0
    episode_start = 0.0
    last_b = last_y = False
    last_img: np.ndarray | None = None
    # Command the arm at --control-freq; sample the dataset every `decim`
    # ticks so episodes land at exactly --freq.
    decim = max(1, int(round(args.control_freq / args.freq)))
    tick = 0
    period = 1.0 / args.control_freq
    interrupted = False
    pending: list[BsplineEpisodeWriter] = []

    try:
        next_tick = time.perf_counter()
        while True:
            b = teleop.is_pause_pressed()     # right B (level)
            y = teleop.is_reverse_pressed()   # left Y (level)

            if b and not last_b:
                if writer is not None:
                    logger.info("✗ episode DISCARDED (%d frames, %.1fs) — restarting now",
                                len(writer), time.perf_counter() - episode_start)
                writer = BsplineEpisodeWriter(output_dir, args.freq)
                tick = 0
                episode_start = time.perf_counter()
                logger.info("● episode STARTED at %s -> %s",
                            time.strftime("%H:%M:%S"), writer.episode_dir.name)

            if y and not last_y and writer is not None:
                if len(writer) == 0:
                    logger.warning("Y pressed with 0 frames recorded — still recording")
                else:
                    dur = time.perf_counter() - episode_start
                    total_frames += len(writer)
                    saved_episodes += 1
                    logger.info("✓ episode SAVED (%d frames, %.1fs, %.1f Hz) -> %s",
                                len(writer), dur, len(writer) / dur if dur > 0 else 0.0,
                                writer.episode_dir)
                    # Encode off the control thread, exactly as their main.py does.
                    writer.flush_async()
                    pending.append(writer)
                    writer = None
                    if args.num_episodes and saved_episodes >= args.num_episodes:
                        logger.info("reached --num-episodes=%d — ending session",
                                    args.num_episodes)
                        break
            last_b, last_y = b, y

            # Their main.py ordering: observe, act, execute, record — but the
            # first three run every control tick (200 Hz) and only the last is
            # decimated to the dataset rate. One get_observations() per tick,
            # reused for the haptic feedback and the recorded frame, so both
            # describe the same instant and the CAN load matches the plain
            # teleop example.
            robot_obs = robot.get_observations()

            action_dict = teleop.get_action()
            _command_arm(robot, action_dict, f"{hand}_")
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
            if writer is not None and tick % decim == 0:
                joints = np.asarray(robot_obs["joint_pos"][:ARM_DOFS], dtype=np.float64)
                gripper = 1.0 - float(robot_obs["gripper_pos"][0])  # i2rt 0=closed -> 0=open
                arm_pos, arm_quat = fk.pose(joints)
                cmd_joints = [float(action_dict[f"{hand}_joint_{j + 1}.pos"])
                              for j in range(ARM_DOFS)]
                cmd_pos, cmd_quat = fk.pose(cmd_joints)
                last_img = _wrist_frame(camera, last_img)
                writer.step(
                    {"arm_pos": arm_pos,
                     "arm_quat": arm_quat,
                     "gripper_pos": np.array([gripper], dtype=np.float64),
                     "wrist_image": last_img},
                    {"arm_pos": cmd_pos,
                     "arm_quat": cmd_quat,
                     "gripper_pos": np.array(
                         [float(action_dict[f"{hand}_gripper.pos"])], dtype=np.float64)},
                )

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
        if writer is not None:
            logger.info("open episode discarded (session ended mid-episode)")
        for w in pending:
            w.wait_for_flush()
        elapsed = time.time() - session_start
        logger.info("session done: %d episode(s), %d frames, %s elapsed — %s",
                    saved_episodes, total_frames,
                    time.strftime("%H:%M:%S", time.gmtime(elapsed)), output_dir.resolve())

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
            if camera is not None:
                camera.stop()
            robot.close()


if __name__ == "__main__":
    main()
