"""Record end-effector-pose-only LeRobot datasets from bimanual YAM VR teleop.

Same session flow as `record_bi_yam.py` — right B starts an episode, left Y
saves it, cameras and `--push-to-hub` behave identically — but what lands in
the dataset is Cartesian:

    observation.state   per arm: [ee_pos(3), ee_rot(6|4), gripper(1)]
                        FK of the arm's MEASURED joints (what the robot did)
    action              per arm: [ee_pos(3), ee_rot(6|4), gripper(1)]
                        FK of the COMMANDED joints (what teleop asked for)

Nothing about how the robot is driven changes: the Quest teleop still solves
IK and streams joint positions to i2rt, exactly as in `record_bi_yam.py`.
This script only changes what is written to disk. The VR stack is already
EE-native internally (the clutch mapper produces an absolute EE target in the
arm-base frame and `DecoupledIKSolver` turns it into joints); recording the
Cartesian pose just stops throwing that representation away.

Why FK of the commanded joints rather than the mapper's raw target: the raw
target can sit ahead of the arm whenever the IK clips it (workspace boundary,
joint limits, per-tick Δq caps), so a policy trained on it would learn to
command poses the arm cannot reach. FK of the commanded joints is always
defined (engaged or not — a disengaged arm holds, so the action holds too),
matches the joint action `record_bi_yam.py` writes frame-for-frame, and
replays cleanly back through IK.

Conventions:

  * Frame — each arm's pose is in THAT ARM'S OWN base frame (the IK builds one
    MuJoCo model per arm, world = arm base). The left→right base transform is
    not modelled anywhere in this repo, so a bimanual dataset holds two
    independent frames. Single-arm (`--arm left|right`) is unambiguous.
  * Point — the `tool0` site: between the fingertips of the linear_4310
    gripper (see `ik/model.py`).
  * Rotation — `--rotation rot6d` (default) writes the first two ROWS of the
    rotation matrix, the pytorch3d / diffusion_policy `rotation_6d`
    convention, so the columns drop straight into that training stack.
    `--rotation quat` writes xyzw with the w >= 0 hemisphere enforced.
  * Gripper — 0 = open, 1 = closed (teleop convention, the inverse of i2rt's).

Deployment note: a policy trained on these actions emits EE poses, and this
repo has no EE-action follower yet — you need a loop that holds the latest
Cartesian target and steps `DecoupledIKSolver.solve()` toward it at a rate
well above the policy's, seeded from the measured joints. `solve()` is a
single damped-least-squares step sized for teleop-rate deltas; feeding it
30 Hz policy output directly will lag on any large jump.

Run (hardware):
    python examples/record_bi_ee_only.py \\
        --repo-id you/yam-ee-task --task "fold the towel" \\
        --left-can can_left --right-can can_right \\
        --num-episodes 20 --push-to-hub

Rehearse the whole flow without motors (i2rt sim robots):
    python examples/record_bi_ee_only.py \\
        --repo-id local/ee-rehearsal --sim --overwrite
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
import mujoco
import numpy as np

from vr_teleop_kit.ik.decoupled_ik import DecoupledIKSolver
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

# Robot-facing helpers are shared with the plain teleop example; the session
# scaffolding (cameras, episode discard, Hub push) with the joint-space
# recorder. Only the feature encoding below is new.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teleop_bi_yam import ARM_DOFS, _command_arm, _gripper_feedback, ramp_to_rest  # noqa: E402
from record_bi_yam import (  # noqa: E402
    DEFAULT_REPO_OWNER,
    _action_vector as _joint_action_vector,
    _discard_open_episode,
    _feature_names as _joint_feature_names,
    _open_cameras,
    _push_to_hub,
)


def _mat_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    flat = np.zeros(9)
    mujoco.mju_quat2Mat(flat, np.asarray(quat_wxyz, dtype=float))
    return flat.reshape(3, 3)


def _rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """First two ROWS of the rotation matrix.

    This is pytorch3d's `matrix_to_rotation_6d` (`matrix[..., :2, :]`), which
    is what diffusion_policy's `RotationTransformer` and the bspline stack's
    `rotation_6d_to_matrix` both assume. Continuous — no quaternion double
    cover — which is why it is the default for the learned action space.
    """
    return _mat_from_quat_wxyz(quat_wxyz)[:2, :].reshape(6)


def _quat_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo wxyz → scipy/robomimic xyzw, restricted to the w >= 0
    hemisphere so q and -q (the same rotation) don't alternate between
    frames and hand the policy a discontinuity that isn't there."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    out = np.array([x, y, z, w])
    return -out if w < 0.0 else out


# name → (encoder, per-component feature-name suffixes)
ROTATION_ENCODINGS = {
    "rot6d": (_rot6d, [f"rot6d_{i + 1}" for i in range(6)]),
    "quat": (_quat_xyzw, ["quat_x", "quat_y", "quat_z", "quat_w"]),
}


def _ee_vector(fk: DecoupledIKSolver, joints, gripper: float, encode_rot) -> list[float]:
    """One arm's [pos(3), rot(6|4), gripper(1)] from its 6 joint angles."""
    pos, quat_wxyz = fk.fk(np.asarray(joints, dtype=float))
    return [*pos, *encode_rot(quat_wxyz), float(np.clip(gripper, 0.0, 1.0))]


def _measure(hands: tuple[str, ...], robots: dict) -> dict[str, dict]:
    """One `get_observations()` per arm per tick. Everything recorded for this
    frame is derived from this single snapshot, so `observation.state` and (with
    `--with-joints`) `measured_joints` describe the same instant — two separate
    reads would sample an arm that kept moving in between."""
    return {hand: robots[hand].get_observations() for hand in hands}


def _obs_ee_vector(hands: tuple[str, ...], measured: dict[str, dict], fk, encode_rot) -> np.ndarray:
    """Measured EE pose per arm, left first. Gripper converted out of i2rt's
    normalized convention (0 = closed) into the teleop's (0 = open)."""
    out: list[float] = []
    for hand in hands:
        obs = measured[hand]
        out.extend(_ee_vector(
            fk, obs["joint_pos"][:ARM_DOFS], 1.0 - float(obs["gripper_pos"][0]), encode_rot))
    return np.asarray(out, dtype=np.float32)


def _obs_joint_vector(hands: tuple[str, ...], measured: dict[str, dict]) -> np.ndarray:
    """Same layout as `record_bi_yam._obs_vector` — per arm [joint1..6, gripper]
    — but read off the snapshot above instead of polling the robot again."""
    out: list[float] = []
    for hand in hands:
        obs = measured[hand]
        out.extend(float(v) for v in obs["joint_pos"][:ARM_DOFS])
        out.append(1.0 - float(obs["gripper_pos"][0]))
    return np.asarray(out, dtype=np.float32)


def _action_ee_vector(action: dict[str, float], hands: tuple[str, ...], fk, encode_rot) -> np.ndarray:
    """Commanded EE pose per arm — FK of the joints this tick's teleop action
    sends to the robot, so the Cartesian action and the joints actually
    commanded describe the same motion."""
    out: list[float] = []
    for hand in hands:
        joints = [float(action[f"{hand}_joint_{j + 1}.pos"]) for j in range(ARM_DOFS)]
        out.extend(_ee_vector(fk, joints, float(action[f"{hand}_gripper.pos"]), encode_rot))
    return np.asarray(out, dtype=np.float32)


def _feature_names(hands: tuple[str, ...], rot_names: list[str]) -> list[str]:
    names: list[str] = []
    for hand in hands:
        names.extend(f"{hand}_ee_pos_{axis}" for axis in ("x", "y", "z"))
        names.extend(f"{hand}_ee_{n}" for n in rot_names)
        names.append(f"{hand}_gripper")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=None,
                    help="dataset repo id (e.g. you/yam-ee-towels). "
                         "Default: atharva/yam-ee-<YYYYmmdd-HHMMSS>")
    ap.add_argument("--task", default="teleop",
                    help="task string stored with every frame (default: 'teleop')")
    ap.add_argument("--root", default=None, help="local dataset root (default: LeRobot's)")
    ap.add_argument("--overwrite", action="store_true",
                    help="if the dataset dir already exists, delete it and start fresh")
    ap.add_argument("--fps", type=int, default=30,
                    help="dataset + control-loop rate (default: 30; the "
                         "tidybot2/diffusion-policy EE stacks run at 10)")
    ap.add_argument("--rotation", choices=tuple(ROTATION_ENCODINGS), default="rot6d",
                    help="EE rotation encoding: rot6d (default, continuous, "
                         "pytorch3d/diffusion_policy convention) or quat (xyzw, w>=0)")
    ap.add_argument("--with-joints", action="store_true",
                    help="also store the measured/commanded joint vectors under the "
                         "'measured_joints'/'commanded_joints' keys. These are NOT "
                         "picked up by LeRobot's policy feature mapping (it only "
                         "classifies 'observation.*'/'action*' keys), so the learned "
                         "spaces stay EE-only — they are there for replay, safety "
                         "review, and re-deriving poses offline.")
    ap.add_argument("--num-episodes", type=int, default=0,
                    help="end the session after this many saved episodes "
                         "(0 = keep going until Ctrl-C)")
    ap.add_argument("--left-can", default="can1", help="left arm CAN interface (default: can1)")
    ap.add_argument("--right-can", default="can0", help="right arm CAN interface (default: can0)")
    ap.add_argument("--arm", choices=("both", "left", "right"), default="both")
    ap.add_argument("--sim", action="store_true",
                    help="use i2rt sim robots (rehearse the recording flow, no hardware)")
    ap.add_argument("--no-cameras", action="store_true",
                    help="skip camera capture (record state/action only). Do not "
                         "enable the Quest camera stream while recording with "
                         "cameras — the relay and recorder can't share a v4l2 device.")
    ap.add_argument("--push-to-hub", action="store_true",
                    help="upload the dataset to the Hugging Face Hub when the "
                         "session ends. Public unless --private is also passed.")
    ap.add_argument("--private", action="store_true",
                    help="with --push-to-hub, create the Hub repo private")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws", help="relay server WS URL")
    ap.add_argument("--rest-duration-s", type=float, default=3.0)
    ap.add_argument("--rest-steps", type=int, default=90)
    add_ik_cli_args(ap)
    args = ap.parse_args()

    setup_logging(level=logging.INFO)
    logger = get_logger(__name__, "session")

    session_start = time.time()
    session_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(session_start))
    if args.repo_id is None:
        args.repo_id = f"{DEFAULT_REPO_OWNER}/yam-ee-{session_stamp}"
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

    hands: tuple[str, ...] = ("left", "right") if args.arm == "both" else (args.arm,)
    channels = {"left": args.left_can, "right": args.right_can}
    encode_rot, rot_names = ROTATION_ENCODINGS[args.rotation]

    # Dedicated FK instance: the teleop's per-arm solvers share their MuJoCo
    # `data` between fk(), solve() and j4_anchor_xpos(), so FK-ing the
    # measured joints through one of those mid-tick would perturb the anchor
    # the clutch reads. Both arms are the same model, so one instance serves
    # both — poses come out in each arm's own base frame either way. Built
    # before the dataset dir so a missing YAM_XML fails fast.
    fk = DecoupledIKSolver()

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

    cameras, cam_features = _open_cameras(not args.no_cameras, hands, logger)
    use_videos = bool(cam_features)

    names = _feature_names(hands, rot_names)
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
        **cam_features,
    }
    if args.with_joints:
        joint_names = _joint_feature_names(hands)
        for key in ("measured_joints", "commanded_joints"):
            features[key] = {
                "dtype": "float32", "shape": (len(joint_names),), "names": joint_names,
            }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,
        robot_type="bi_yam_ee" if args.arm == "both" else "yam_ee",
        use_videos=use_videos,
        image_writer_threads=4 * len(cameras),
        features=features,
    )
    logger.info("dataset %s — LeRobot codebase %s, %d fps, %d-dim EE state/action "
                "(%s), cameras=%s, videos=%s",
                args.repo_id, CODEBASE_VERSION, args.fps, len(names), args.rotation,
                list(cameras) or "none", use_videos)
    logger.info("EE pose = tool0 site (between the fingertips), gripper 0=open 1=closed")
    if len(hands) == 2:
        logger.warning(
            "bimanual: left and right EE poses are each in THEIR OWN arm-base "
            "frame — the base-to-base transform is not modelled in this repo. "
            "A policy trained on this data learns two independent frames; if you "
            "need a shared frame, measure the mount transform and compose it here.")
    logger.info("session started %s — writing to %s",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_start)), dataset.root)

    rests = {
        "left": parse_rest_pose_env("LEFT_REST_POSE", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "right": parse_rest_pose_env("RIGHT_REST_POSE", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    robots = {h: get_yam_robot(channel=channels[h], sim=args.sim) for h in hands}

    teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(
        id="vr-record-ee",
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

            # lerobot-record ordering: observe, act, command, write. The arms
            # are still driven in joint space — only the recorded features are
            # Cartesian.
            measured = _measure(hands, robots)
            obs_ee = _obs_ee_vector(hands, measured, fk, encode_rot)
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
                    "observation.state": obs_ee,
                    "action": _action_ee_vector(action, hands, fk, encode_rot),
                    "task": args.task,
                }
                if args.with_joints:
                    frame["measured_joints"] = _obs_joint_vector(hands, measured)
                    frame["commanded_joints"] = _joint_action_vector(action, hands)
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

        # First Ctrl-C parks the arms and holds them; a second powers off.
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
