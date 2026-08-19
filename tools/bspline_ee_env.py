"""Deploy an EE-space policy trained on vr-teleop-kit data, with matching kinematics.

Drop-in replacement for bspline-policy's `RealEnv` for **end-effector** policies
(`single_yam_rot6d`). Same interface — `get_obs()`, `step()`, `reset()`,
`close()`, `.arm` — so `rollout_local_policy.py` uses it unchanged.

Why this exists
---------------
The EE dataset's `arm_pos` / `arm_quat` were computed from *this repo's* MuJoCo
model. `yam_server` does FK and IK with pyroki against *its own* URDF, and the
two chains disagree by up to ~9 mm at the same joint angles — measured in
float64, and not a fixed offset you can subtract away (the tool0→TCP transform
varies by ~1 cm across configurations).

Deploying straight through `yam_server` therefore applies that error twice:
the policy is handed an observation pose in a frame it never trained on, and
its Cartesian output is converted back to joints through the same wrong map.

This wrapper avoids both by using `yam_server` purely as a **joint-level motor
driver**, which is model-agnostic:

    get_obs()   arm_pos / arm_quat recomputed from the measured `joint_pos`
                with this repo's FK, in the TCP convention the dataset used.
                Images and gripper pass through untouched.

    step()      the policy's Cartesian target goes through `EEFollower`
                (this repo's IK) and is sent as `{"joint_pos": ...}`, which
                yam_server executes directly — `_joint_mode = True` bypasses
                its own IK.

Everything the policy sees and everything the arm executes is then in the same
kinematic frame as training.

Usage — add one branch to `rollout_local_policy.make_env`:

    if args.env == "yam-vrkit":
        sys.path.insert(0, "/home/yambox/vr-teleop-kit/tools")
        from bspline_ee_env import BsplineEEEnv
        return BsplineEEEnv(
            use_cameras=not args.no_cameras,
            stiffness_kp_scale=args.stiffness_kp_scale,
            control_freq=args.control_freq,
        )

then run the normal rollout with `--env yam-vrkit --speed-up-times 1.0`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

_VRKIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_VRKIT_ROOT / "src"))

from vr_teleop_kit.ik.decoupled_ik import DecoupledIKSolver  # noqa: E402
from vr_teleop_kit.ik.ee_follower import EEFollower  # noqa: E402

logger = logging.getLogger(__name__)

# The arm MJCF the dataset's poses were computed from. Pinned to THIS repo's
# i2rt clone: bspline-policy ships a different yam.xml (its link_6 body has a
# different origin), and silently loading that one would reintroduce exactly
# the frame error this wrapper exists to remove.
DEFAULT_MODEL_PATH = _VRKIT_ROOT / "i2rt" / "i2rt" / "robot_models" / "arm" / "yam" / "yam.xml"


class BsplineEEEnv:
    """`RealEnv` with vr-teleop-kit kinematics on both the obs and action side."""

    def __init__(
        self,
        use_cameras: bool = True,
        stiffness_kp_scale: float = 1.0,
        control_freq: float = 100.0,
        mu: float = 0.0,
        pos_reach_limit: float = 0.25,
        rot_reach_limit: float = 0.6,
        gripper_max_speed: float = 3.0,
        model_path: str | Path | None = None,
    ) -> None:
        from real_env import RealEnv  # bspline-policy; caller has it on sys.path

        self._env = RealEnv(use_cameras=use_cameras, stiffness_kp_scale=stiffness_kp_scale)

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"YAM model not found at {path}. This must be the SAME arm MJCF the "
                "dataset was recorded with (vr-teleop-kit's i2rt clone), not "
                "bspline-policy's copy — they differ.")
        # mu low by design: the solver's posture-bias term is a steady-state
        # offset against a fixed target (3.4 mm at the teleop default 0.02).
        # Under teleop the operator corrects it; a policy cannot.
        self._solver = DecoupledIKSolver(mu=mu, model_path=path)

        state = self._env.arm.get_state()
        q0, g0 = self._split_joint_pos(state)
        self.follower = EEFollower(
            self._solver,
            freq=control_freq,
            target_frame="tcp",  # the dataset's flange convention, 13.47 cm behind tool0
            q_init=q0,
            gripper_init=g0,
            pos_reach_limit=pos_reach_limit,
            rot_reach_limit=rot_reach_limit,
            gripper_max_speed=gripper_max_speed,
        )
        logger.info("BsplineEEEnv: model=%s mu=%s control_freq=%.0f Hz frame=tcp",
                    path, mu, control_freq)
        print(f"[BsplineEEEnv] kinematics from {path}\n"
              f"[BsplineEEEnv] IK mu={mu}, {control_freq:.0f} Hz, target_frame=tcp, "
              f"reach limits {pos_reach_limit} m / {rot_reach_limit} rad")

    # ---------- helpers ----------

    @staticmethod
    def _split_joint_pos(state: dict) -> tuple[np.ndarray, float]:
        if "joint_pos" not in state:
            raise RuntimeError(
                "yam_server.get_state() returned no 'joint_pos' — this wrapper needs "
                "the joint-space state (update yam_server.py).")
        jp = np.asarray(state["joint_pos"], dtype=np.float64).reshape(-1)
        if jp.size != 7:
            raise RuntimeError(f"joint_pos must be 7D [joint1..6, gripper], got {jp.size}")
        return jp[:6].copy(), float(jp[6])

    # ---------- RealEnv interface ----------

    @property
    def arm(self):
        return self._env.arm

    def get_cameras(self):
        return self._env.get_cameras()

    def get_obs(self) -> dict:
        """Their obs, with arm_pos/arm_quat recomputed in our frame."""
        obs = self._env.get_obs()
        q, _ = self._split_joint_pos(obs)
        # FK of the MEASURED joints, in the dataset's TCP convention. Overwrites
        # the server's pyroki values — same quantity, our kinematic chain.
        self.follower._q = q  # noqa: SLF001 — read-only use for the FK below
        pos, quat_xyzw = self.follower.ee_pose("tcp")
        obs["arm_pos"] = pos
        obs["arm_quat"] = quat_xyzw
        return obs

    def step(self, action: dict) -> None:
        """Cartesian target -> one IK step -> joint command."""
        if "joint_pos" in action:  # a joint-space policy: pass straight through
            self._env.step(action)
            return

        gripper = float(np.asarray(action["gripper_pos"]).reshape(-1)[0])
        self.follower.set_target(action["arm_pos"], action["arm_quat"], gripper)
        q, grip = self.follower.step()
        self._env.step({"joint_pos": np.concatenate([q, [grip]])})

    def reset(self) -> None:
        self._env.reset()
        # Re-seed the virtual joint state from where the arm actually parked,
        # or the first IK step would integrate from a stale configuration.
        q0, g0 = self._split_joint_pos(self._env.arm.get_state())
        self.follower._q = q0            # noqa: SLF001
        self.follower._gripper = g0      # noqa: SLF001
        self.follower._target_pos = None  # noqa: SLF001 — hold until the first action
        self.follower._target_quat_wxyz = None  # noqa: SLF001
        self.follower._target_gripper = g0      # noqa: SLF001

    def close(self, *args, **kwargs):
        return self._env.close(*args, **kwargs)

    def __getattr__(self, name):
        # Anything else the rollout pokes at (speed scaling, stiffness, ...)
        # falls through to the wrapped env.
        return getattr(self._env, name)
