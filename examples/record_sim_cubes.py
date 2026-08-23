"""Teleop and record the cube-stacking scene in MuJoCo, driven from the Quest.

Same controls and same dataset schema as ``record_bi_yam.py``, but the robot is
``sim/spd_scene.xml`` -- both YAMs, the measured table, and three PLA cubes --
instead of hardware. Rendered camera frames are published to the relay over
shared memory, so the headset shows the simulated scene through the same WebRTC
path it uses for real cameras.

This is *not* ``record_bi_yam.py --sim``. That flag uses i2rt's ``SimRobot``,
which loads a single-arm XML per arm: two arms in two separate worlds, with no
table and nothing to pick up. Stacking needs both arms and the objects in one
model, which is what this drives.

Run it in three pieces:

    # 1. relay, told to serve simulated cameras instead of v4l2
    VR_TELEOP_SIM_FRAMES=1 vr-teleop-relay

    # 2. this, which creates the shared-memory bus and connects to the relay
    DISPLAY=:0 MUJOCO_GL=glfw python examples/record_sim_cubes.py \\
        --repo-id atharva/sim-cubes --task "stack the cubes" --num-episodes 3

    # 3. Quest browser -> http://localhost:8443/ , enable the camera stream

    # optional: live 3D view of the running session, in any browser
    python tools/sim_viser.py --live --port 8000

Start order does not matter: the relay attaches to the bus lazily on the first
WebRTC request and warns if the simulator is not up yet.

In-VR controls (identical to the hardware recorder):

    grip (squeeze)      drive that arm, trigger = gripper
    right B (idle)      start an episode
    right B (recording) throw the take away and restart it
    left Y              save the episode

Episodes reset to the scene's `start` keyframe -- the pose real demos begin
from -- so the arms start folded back near the near edge, as they do on the
robot, and you raise them yourself.

**20 fps, not 30.** Each offscreen ``render()`` costs ~11 ms of GL readback on
this machine regardless of resolution -- it is per-call overhead, not pixels --
so three cameras cost ~33 ms, which does not fit a 30 Hz tick but sits
comfortably inside 20 Hz. Physics still runs at the scene's 480 Hz.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from vr_teleop_kit.log import get_logger, setup_logging  # noqa: E402
from vr_teleop_kit.lerobot.bi_quest_teleop import (  # noqa: E402
    BiQuestTeleoperator,
    BiQuestTeleoperatorConfig,
)
from vr_teleop_kit.lerobot.cli import (  # noqa: E402
    add_ik_cli_args,
    ik_kwargs_from_args,
)
from vr_teleop_kit.relay.sim_frames import CAM_IDS, SimFrameBus  # noqa: E402
from vr_teleop_kit.ik.decoupled_ik import DecoupledIKSolver  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the end-effector encoders from the hardware EE recorder rather than
# re-deriving them, so a sim EE dataset and a real one are byte-compatible.
from record_bi_ee_only import (  # noqa: E402
    ROTATION_ENCODINGS,
    _ee_vector,
    _feature_names as _ee_feature_names,
)

ARM_DOFS = 6
HANDS = ("left", "right")
# Blocks whose pose is recorded alongside the robot state. This is *privileged*
# information -- a vision policy should never be fed it -- but without it a
# recording cannot be scored after the fact: joint angles and camera frames
# alone do not say whether a stack actually formed, only how many times the
# operator closed the gripper. Recording it makes every episode objectively
# checkable ("did the blocks end up stacked, and how square").
BLOCKS = ("cube_left", "cube_centre", "cube_right")
# Per-block spawn jitter: each block is placed anywhere in an 8 x 8 cm region
# centred on its nominal spot, so the three can end up in genuinely different
# relative arrangements between episodes (the paper randomises "initial
# positions of all objects" on every reset).
#
# With the row centre at 0.44 m the region spans 0.40-0.48 m from the arm bases.
# That reaches back into the badly-conditioned part of the workspace: measured at
# the centre stack point, condition number runs 22 at 0.40 m and 196 at 0.45 m,
# with joints 4 and 6 pinned on their limits beyond 0.40 m. A block spawning at
# the far edge may be awkward or impossible to stack -- 0.35 m was chosen
# earlier precisely to avoid this.
SPAWN_REGION_M = 0.08
# Colour is assigned by final left-to-right order, not by body: whichever block
# ends up leftmost is the black one, the other two red. So the *semantics* stay
# fixed while the arrangement varies. Tint jitter follows the paper's "randomly
# tint object colors", kept inside each colour's family so black stays black.
BLOCK_BASE_RGB = {"black": (0.05, 0.05, 0.05), "red": (0.70, 0.06, 0.06)}
TINT_JITTER = 0.06
BLOCK_FIELDS = ("x", "y", "z", "qw", "qx", "qy", "qz")
# Measured jaw travel: joint7/joint8 at 0 is closed, 0.0475 is fully open
# (~95 mm between the finger faces). The teleop's gripper.pos runs the other
# way, 0 = open .. 1 = closed, hence the flip in _apply_action.
GRIPPER_OPEN_QPOS = 0.0475
DEFAULT_REPO_OWNER = "atharva"


def _feature_names() -> list[str]:
    return [f"{hand}_{n}" for hand in HANDS
            for n in [f"joint_{j+1}" for j in range(ARM_DOFS)] + ["gripper"]]


def _block_feature_names() -> list[str]:
    return [f"{b}_{f}" for b in BLOCKS for f in BLOCK_FIELDS]


class SimRig:
    """The MuJoCo scene plus the bits of robot API the teleop loop needs."""

    def __init__(self, scene: str, fps: int, logger) -> None:
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.log = logger
        self.substeps = max(1, round((1.0 / fps) / self.model.opt.timestep))
        # Episodes open from the pose real demonstrations open from, not the
        # all-zero mechanical rest pose and not a hand-picked "looks good in the
        # camera" pose. sim/build_scene.py builds it from the measured median
        # first frame of the real training episodes.
        self.key_ready = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "start")
        if self.key_ready < 0:
            self.key_ready = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")

        self.jadr: dict[str, int] = {}
        self.actid: dict[str, int] = {}
        for hand in HANDS:
            for j in range(1, 9):
                name = f"{hand}_joint{j}"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    self.jadr[name] = self.model.jnt_qposadr[jid]
            for a in [f"{hand}_joint{j}" for j in range(1, 7)] + [f"{hand}_gripper"]:
                self.actid[a] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)

        self.cube_joints = {}
        for cube in BLOCKS:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{cube}_free")
            if jid >= 0:
                self.cube_joints[cube] = (self.model.jnt_qposadr[jid],
                                          self.model.jnt_dofadr[jid])
        self.cube_home = {}
        self.cube_geom = {c: mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{c}_geom") for c in BLOCKS}
        self.fk = DecoupledIKSolver()

        self.reset()
        # Remember where the cubes start, to restore between episodes.
        for cube, (adr, _) in self.cube_joints.items():
            self.cube_home[cube] = self.data.qpos[adr:adr + 7].copy()

        self.cams = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, c)
                     for c in CAM_IDS]
        self.renderer = mujoco.Renderer(self.model, 480, 640)
        logger.info("scene %s: nq=%d, %d substeps/tick (%.0f Hz physics)",
                    scene, self.model.nq, self.substeps,
                    1.0 / self.model.opt.timestep)

    # ── state ───────────────────────────────────────────────────────────
    def reset(self) -> None:
        if self.key_ready >= 0:
            self.mj.mj_resetDataKeyframe(self.model, self.data, self.key_ready)
        else:
            self.mj.mj_resetData(self.model, self.data)
        self.hold_current()
        self.mj.mj_forward(self.model, self.data)

    def reset_cubes(self, rng: np.random.Generator | None = None,
                    region_m: float = SPAWN_REGION_M) -> None:
        """Re-spawn the blocks, jittered and re-coloured, arms untouched.

        Positions are drawn inside a `region_m` square per block. Colour is then
        assigned by the resulting left-to-right order, so the leftmost block is
        always the black one however the arrangement came out -- the task stays
        legible while the layout varies.
        """
        placed: dict[str, float] = {}
        for cube, (adr, dof) in self.cube_joints.items():
            home = self.cube_home[cube].copy()
            if rng is not None and region_m > 0:
                half = region_m / 2.0
                home[0] += rng.uniform(-half, half)
                home[1] += rng.uniform(-half, half)
            self.data.qpos[adr:adr + 7] = home
            self.data.qvel[dof:dof + 6] = 0.0
            placed[cube] = float(home[0])

        order = sorted(placed, key=lambda c: placed[c])
        for i, cube in enumerate(order):
            base = BLOCK_BASE_RGB["black" if i == 0 else "red"]
            rgb = np.array(base, dtype=float)
            if rng is not None:
                rgb = np.clip(rgb + rng.uniform(-TINT_JITTER, TINT_JITTER, 3), 0.02, 1.0)
            gid = self.cube_geom.get(cube)
            if gid is not None and gid >= 0:
                self.model.geom_rgba[gid, :3] = rgb
        self.mj.mj_forward(self.model, self.data)
        return {c: (round(placed[c], 4),
                    "black" if c == order[0] else "red") for c in BLOCKS}

    def ee_measured(self, encode_rot) -> np.ndarray:
        """End-effector pose implied by the CURRENT joint state.

        Must be sampled at the same instant as obs_vector(): both describe the
        same frame, and if one is read before the physics step and the other
        after, the two representations of that frame disagree. That bug put
        observation.ee a full tick behind observation.state -- invisible in any
        shape or finiteness check, and worth up to a whole unit of gripper
        travel whenever the jaws moved during the tick.

        Poses come out in each arm's own base frame, matching what the hardware
        EE recorder stores, so a sim EE dataset lines up with a real one.
        """
        out = []
        for hand in HANDS:
            q = [float(self.data.qpos[self.jadr[f"{hand}_joint{j+1}"]])
                 for j in range(ARM_DOFS)]
            grip = 1.0 - float(np.clip(
                self.data.qpos[self.jadr[f"{hand}_joint7"]] / GRIPPER_OPEN_QPOS,
                0.0, 1.0))
            out.extend(_ee_vector(self.fk, q, grip, encode_rot))
        return np.asarray(out, dtype=np.float32)

    def ee_commanded(self, action: dict[str, float], encode_rot) -> np.ndarray:
        """End-effector pose implied by the commanded joint action.

        A pure function of the action, so it is order-independent.
        """
        out = []
        for hand in HANDS:
            q = [float(action[f"{hand}_joint_{j+1}.pos"]) for j in range(ARM_DOFS)]
            out.extend(_ee_vector(self.fk, q,
                                  float(action[f"{hand}_gripper.pos"]), encode_rot))
        return np.asarray(out, dtype=np.float32)

    def hold_current(self) -> None:
        """Command every joint to hold wherever it currently is."""
        for hand in HANDS:
            for j in range(1, 7):
                a = self.actid[f"{hand}_joint{j}"]
                if a >= 0:
                    self.data.ctrl[a] = self.data.qpos[self.jadr[f"{hand}_joint{j}"]]
            a = self.actid[f"{hand}_gripper"]
            if a >= 0:
                self.data.ctrl[a] = self.data.qpos[self.jadr[f"{hand}_joint7"]]

    def obs_pos_dict(self) -> dict[str, float]:
        """``.pos``-keyed dict in the teleop's conventions (0=open..1=closed)."""
        out: dict[str, float] = {}
        for hand in HANDS:
            for j in range(ARM_DOFS):
                out[f"{hand}_joint_{j+1}.pos"] = float(
                    self.data.qpos[self.jadr[f"{hand}_joint{j+1}"]])
            grip = float(self.data.qpos[self.jadr[f"{hand}_joint7"]])
            out[f"{hand}_gripper.pos"] = 1.0 - float(
                np.clip(grip / GRIPPER_OPEN_QPOS, 0.0, 1.0))
        return out

    def obs_vector(self) -> np.ndarray:
        """Per arm: [joint1..6, gripper], left first -- matching the hardware
        recorder's observation.state layout exactly, so sim and real datasets
        are directly comparable."""
        o = self.obs_pos_dict()
        return np.asarray(
            [v for h in HANDS
             for v in [o[f"{h}_joint_{j+1}.pos"] for j in range(ARM_DOFS)]
                      + [o[f"{h}_gripper.pos"]]],
            dtype=np.float32)

    def block_state(self) -> np.ndarray:
        """Every block's pose as [x, y, z, qw, qx, qy, qz], in block order.

        Read from qpos rather than xpos/xquat so it is exactly the freejoint
        state -- the same numbers a reset writes back, which makes an episode
        replayable from its own recording.
        """
        out: list[float] = []
        for cube in BLOCKS:
            adr = self.cube_joints.get(cube, (None, None))[0]
            if adr is None:
                out.extend([0.0] * len(BLOCK_FIELDS))
            else:
                out.extend(float(v) for v in self.data.qpos[adr:adr + 7])
        return np.asarray(out, dtype=np.float32)

    def gripper_feedback(self) -> dict[str, float]:
        """Gripper effort + position, for the Quest's force haptics.

        The sim has a real actuator force to report, so haptics work here the
        same way they do on hardware -- squeezing a cube pushes back.
        """
        out: dict[str, float] = {}
        for hand in HANDS:
            a = self.actid[f"{hand}_gripper"]
            if a >= 0:
                out[f"{hand}_gripper.torque"] = float(self.data.actuator_force[a])
            grip = float(self.data.qpos[self.jadr[f"{hand}_joint7"]])
            out[f"{hand}_gripper.pos"] = 1.0 - float(
                np.clip(grip / GRIPPER_OPEN_QPOS, 0.0, 1.0))
        return out

    # ── control ─────────────────────────────────────────────────────────
    def apply_action(self, action: dict[str, float]) -> None:
        for hand in HANDS:
            for j in range(ARM_DOFS):
                a = self.actid[f"{hand}_joint{j+1}"]
                if a >= 0:
                    self.data.ctrl[a] = float(action[f"{hand}_joint_{j+1}.pos"])
            a = self.actid[f"{hand}_gripper"]
            if a >= 0:
                closed = float(np.clip(action[f"{hand}_gripper.pos"], 0.0, 1.0))
                self.data.ctrl[a] = (1.0 - closed) * GRIPPER_OPEN_QPOS

    def step(self) -> None:
        for _ in range(self.substeps):
            self.mj.mj_step(self.model, self.data)

    def render(self) -> dict[str, np.ndarray]:
        """One RGB frame per camera. Caller converts for the bus/dataset."""
        out = {}
        for cid, cam in zip(CAM_IDS, self.cams):
            if cam < 0:
                continue
            self.renderer.update_scene(self.data, camera=cam)
            out[cid] = self.renderer.render()
        return out


def _action_vector(action: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [v for h in HANDS
         for v in [float(action[f"{h}_joint_{j+1}.pos"]) for j in range(ARM_DOFS)]
                  + [float(action[f"{h}_gripper.pos"])]],
        dtype=np.float32)


def _discard_open_episode(dataset, logger) -> None:
    """Drop the in-progress episode and the temp frames written for it.

    Mirrors record_bi_yam.py: clear_episode_buffer only deletes dirs for
    features with dtype "image", and cameras here are "video", so the raw PNGs
    the async writer dumped would otherwise be left on disk.
    """
    ep_index = dataset.num_episodes
    dataset.clear_episode_buffer()
    writer = getattr(dataset, "writer", None)
    cleanup = getattr(writer, "cleanup_interrupted_episode", None)
    if cleanup is None:
        return
    try:
        cleanup(ep_index)
    except Exception:
        logger.debug("cleanup_interrupted_episode(%d) failed", ep_index,
                     exc_info=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--task", default="stack the cubes")
    ap.add_argument("--root", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing dataset instead of creating "
                         "one. --num-episodes then counts NEW episodes to add, "
                         "not the total.")
    ap.add_argument("--scene", default=str(_REPO_ROOT / "sim" / "spd_scene.xml"))
    ap.add_argument("--fps", type=int, default=20,
                    help="dataset + control rate. 20 because three offscreen "
                         "renders cost ~33 ms here (per-call GL readback, not "
                         "pixels), which does not fit a 30 Hz tick")
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--spawn-region", type=float, default=SPAWN_REGION_M,
                    help="side of the square each block spawns in, in metres "
                         "(0 = identical layout every episode)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for block spawn + tint. Default: drawn from "
                         "the OS, logged and stored in the dataset name's "
                         "metadata so a session is reproducible but every new "
                         "recording differs.")
    ap.add_argument("--rotation", choices=tuple(ROTATION_ENCODINGS), default="rot6d",
                    help="end-effector rotation encoding for the EE features")
    ap.add_argument("--no-cameras", action="store_true",
                    help="skip rendering entirely: state/action only, and the "
                         "headset shows nothing")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws")
    add_ik_cli_args(ap)
    args = ap.parse_args()

    setup_logging(level=logging.INFO)
    logger = get_logger(__name__, "simrec")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.repo_id is None:
        args.repo_id = f"{DEFAULT_REPO_OWNER}/sim-cubes-{stamp}"
        logger.info("no --repo-id given; using %s", args.repo_id)

    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    if args.resume:
        if not root.exists():
            logger.error("--resume given but %s does not exist", root)
            sys.exit(1)
        if args.overwrite:
            logger.error("--resume and --overwrite are contradictory")
            sys.exit(1)
    elif root.exists():
        if args.overwrite:
            logger.warning("--overwrite: removing %s", root)
            shutil.rmtree(root)
        else:
            logger.error("dataset already exists at %s — pass --resume to add "
                         "episodes to it, --overwrite to replace it, or choose "
                         "another --repo-id", root)
            sys.exit(1)

    rig = SimRig(args.scene, args.fps, logger)
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "big")
    rng = np.random.default_rng(seed)
    logger.info("randomisation seed %d (pass --seed %d to reproduce this session)",
                seed, seed)
    encode_rot, rot_names = ROTATION_ENCODINGS[args.rotation]

    bus = None
    cam_features: dict[str, dict] = {}
    if not args.no_cameras:
        bus = SimFrameBus(create=True)
        logger.info("shared-memory frame bus up; run the relay with "
                    "VR_TELEOP_SIM_FRAMES=1 to show it in the headset")
        for cid in CAM_IDS:
            cam_features[f"observation.images.{cid}"] = {
                "dtype": "video", "shape": (480, 640, 3),
                "names": ["height", "width", "channels"]}

    names = _feature_names()
    block_names = _block_feature_names()
    # Both action spaces in one recording. Joint space stays the canonical
    # "action" so this dataset is interchangeable with record_bi_yam.py's; the
    # end-effector pair is carried alongside, encoded exactly as
    # record_bi_ee_only.py does, so an EE dataset can be derived without
    # re-recording. Which space transfers better is an open question, and
    # collecting both costs one FK per arm per tick.
    ee_names = _ee_feature_names(HANDS, rot_names)
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),),
                              "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.blocks": {"dtype": "float32",
                               "shape": (len(block_names),),
                               "names": block_names},
        "observation.ee": {"dtype": "float32", "shape": (len(ee_names),),
                           "names": ee_names},
        "action.ee": {"dtype": "float32", "shape": (len(ee_names),),
                      "names": ee_names},
        **cam_features,
    }
    if args.resume:
        # LeRobotDataset(...) alone opens the dataset READ-ONLY; add_frame then
        # raises. resume() is the API that attaches a writer for appending.
        dataset = LeRobotDataset.resume(
            repo_id=args.repo_id, root=args.root,
            image_writer_threads=4 * max(len(cam_features), 1))
        already = dataset.num_episodes
        logger.info("resuming %s — %d episode(s) already recorded, adding %d more",
                    args.repo_id, already, args.num_episodes)
    else:
        already = 0
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id, fps=args.fps, root=args.root,
            robot_type="bi_yam_sim", use_videos=bool(cam_features),
            image_writer_threads=4 * len(cam_features),
            features=features,
        )
    logger.info("dataset %s (LeRobot %s) at %s — %d fps, %d-dim state/action, "
                "%d-dim block poses, %d-dim EE (%s)",
                args.repo_id, CODEBASE_VERSION, dataset.root, args.fps,
                len(names), len(block_names), len(ee_names), args.rotation)

    teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(
        id="sim-cubes", ws_url=args.ws_url,
        rest_qpos_left=[0.0] * 6, rest_qpos_right=[0.0] * 6,
        **ik_kwargs_from_args(args)))
    teleop.connect()
    # Anchor the IK to the scene's actual pose, not the teleop's rest pose,
    # so the first emitted action does not jolt the arms.
    teleop.seed_qpos_from_obs(rig.obs_pos_dict())

    logger.info("──────────────────────────────────────────────────")
    logger.info("right B: start episode / discard+restart   left Y: save")
    logger.info("session ends after %d newly saved episode(s)", args.num_episodes)
    logger.info("──────────────────────────────────────────────────")

    recording = False
    frames_in_episode = 0
    total_frames = 0
    episode_start = 0.0
    last_b = last_y = False
    period = 1.0 / args.fps
    interrupted = False
    last_frames: dict[str, np.ndarray] = {}


    # LeRobot writes the parquet footer only in finalize(), so a raw SIGTERM
    # (kill, tmux pane close) or SIGHUP (dropped SSH session) would truncate the
    # shard and take every episode in it down with the run. Turn both into a
    # KeyboardInterrupt so the finally-block below still finalizes.
    def _graceful(signum, _frame):
        logger.warning("signal %s received — finalizing dataset before exit",
                       signal.Signals(signum).name)
        raise KeyboardInterrupt
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(_sig, _graceful)
        except (ValueError, OSError):  # not on the main thread / unsupported
            pass

    try:
        next_tick = time.perf_counter()
        while True:
            b = teleop.is_pause_pressed()
            y = teleop.is_reverse_pressed()

            if b and not last_b:
                if recording:
                    dur = time.perf_counter() - episode_start
                    _discard_open_episode(dataset, logger)
                    logger.info("✗ episode DISCARDED (%d frames, %.1fs) — restarting",
                                frames_in_episode, dur)
                layout = rig.reset_cubes(rng, args.spawn_region)
                recording, frames_in_episode = True, 0
                episode_start = time.perf_counter()
                logger.info("● episode %d STARTED — layout %s",
                            dataset.num_episodes,
                            ", ".join(f"{k.split('_')[1]}:x={v[0]:+.3f}/{v[1]}"
                                      for k, v in layout.items()))

            if y and not last_y and recording:
                if frames_in_episode == 0:
                    logger.warning("Y with 0 frames — still recording")
                else:
                    dur = time.perf_counter() - episode_start
                    total_frames += frames_in_episode
                    dataset.save_episode()
                    recording = False
                    logger.info("✓ episode %d SAVED (%d frames, %.1fs) — %d total",
                                dataset.num_episodes - 1, frames_in_episode,
                                dur, total_frames)
                    if dataset.num_episodes - already >= args.num_episodes:
                        logger.info("added %d new episode(s) (%d total) — ending",
                                    dataset.num_episodes - already,
                                    dataset.num_episodes)
                        break
            last_b, last_y = b, y

            # Sample every observation for this frame at the SAME instant,
            # before the action is applied and physics advances.
            obs_vec = rig.obs_vector()
            ee_meas = rig.ee_measured(encode_rot)
            blocks_vec = rig.block_state()
            action = teleop.get_action()
            rig.apply_action(action)
            rig.step()
            try:
                teleop.send_feedback({"torques": rig.gripper_feedback()})
            except Exception:
                pass

            if bus is not None:
                # Publish qpos every tick so `sim_viser.py --live` can show the
                # running session. Cheap: 37 doubles into shared memory.
                bus.write_state(rig.data.qpos)
            frames = rig.render() if bus is not None else {}
            for cid, rgb in frames.items():
                # A reversed view, not a copy: bus.write assigns into shared
                # memory anyway, so .copy() here would be a second 921 KB
                # memcpy per camera per tick for nothing.
                bus.write(cid, rgb[:, :, ::-1])          # bus wants BGR
                last_frames[cid] = rgb

            if recording:
                frame = {
                    "observation.state": obs_vec,
                    "action": _action_vector(action),
                    "observation.blocks": blocks_vec,
                    "observation.ee": ee_meas,
                    "action.ee": rig.ee_commanded(action, encode_rot),
                    "task": args.task,
                }
                for cid in CAM_IDS:
                    img = frames.get(cid, last_frames.get(cid))
                    if img is not None:
                        # LeRobot's writer treats a 3-channel frame as RGB,
                        # which is what the renderer already produces.
                        frame[f"observation.images.{cid}"] = img
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
        logger.info("interrupted")
    finally:
        if recording:
            _discard_open_episode(dataset, logger)
            logger.info("open episode discarded (ended mid-episode)")
        if hasattr(dataset, "finalize"):
            dataset.finalize()
        logger.info("done: %d episode(s), %d frames%s",
                    dataset.num_episodes, total_frames,
                    " (interrupted)" if interrupted else "")
        logger.info("dataset at %s", dataset.root)
        try:
            teleop.disconnect()
        finally:
            if bus is not None:
                bus.close()


if __name__ == "__main__":
    main()
