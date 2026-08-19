"""Drive a YAM toward a stream of Cartesian end-effector targets.

This is the deployment-side counterpart to the teleop loop. Teleop feeds
`DecoupledIKSolver` a target pose derived from the operator's hand; a policy
instead emits EE poses at its own (usually low) rate. `EEFollower` bridges the
two: it holds the most recent target and steps the solver toward it at a fixed
rate well above the policy's.

Why the rate split matters: `solve()` is a single damped-least-squares step
bounded by `max_dq_per_joint`, not a converged IK. Handing it a 10 Hz policy
output directly means one clipped step per command and a visibly lagging arm.
Running it at 100-200 Hz against a latest-wins target reproduces the structure
that both the teleop here and `bspline-policy`'s `yam_server` already use.

Two conventions this class is careful about:

* **Virtual joint state.** The solver is seeded from its own integrated state,
  not from the robot's measured joints. Feeding measurements back couples
  tracking error into the solve and can wind up under load. Seed once at
  startup (`q_init`) and integrate from there — the same choice their
  `yam_server` makes with `_q_pyroki`.

* **Posture bias.** The solver's Tikhonov term (`mu`) pulls joints 1-3 toward
  the rest pose. Under teleop that is invisible — the operator closes the loop
  by eye — but against a fixed target it is a *steady-state offset*: at the
  teleop default `mu=0.02` the arm settles 3.4 mm from the commanded pose, at
  `mu=0.005` 0.22 mm, and at `mu=0.0` it converges exactly. A deployed policy
  specifies the pose itself and has no operator to absorb the bias, so build
  the solver with a low `mu` for rollouts; this class warns if it is high.

* **Target frame.** `solve()` targets the `tool0` site (between the
  fingertips). A policy trained on a bspline/tidybot2-format dataset emits
  poses in *their* TCP convention — the link6 flange re-axed — which sits
  13.47 cm behind tool0. Pass `target_frame="tcp"` for that data and the
  conversion is applied exactly; get it wrong and the arm reaches ~13.5 cm
  past everything. The transform is read off the model at construction rather
  than hardcoded.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import mujoco
import numpy as np

from vr_teleop_kit.core.pose_mapping import (
    quat_conj, quat_mul, quat_to_rotvec, rotvec_to_quat,
)

logger = logging.getLogger(__name__)

# Above this the posture bias costs more than ~1 mm of steady-state accuracy.
MU_DEPLOY_MAX = 0.005

# Fixed link6 -> TCP transform used by bspline-policy's `yam_server`: a pure
# rotation, so their TCP is the flange origin with re-axed orientation.
T_LINK6_TO_TCP = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

TARGET_FRAMES = ("tool0", "tcp")


def _mat_from_quat_xyzw(quat_xyzw) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    wxyz = np.array([q[3], q[0], q[1], q[2]])
    flat = np.zeros(9)
    mujoco.mju_quat2Mat(flat, wxyz)
    return flat.reshape(3, 3)


def _quat_wxyz_from_mat(R: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.ascontiguousarray(R, dtype=np.float64).reshape(9))
    return out


class EEFollower:
    """Step a `DecoupledIKSolver` toward the latest Cartesian target.

    Args:
        solver: a `DecoupledIKSolver` for the arm being driven.
        send_joints: called each tick as ``send_joints(qpos6, gripper)`` with
            the gripper in the teleop convention (0 = open, 1 = closed). Wire
            it to your driver; nothing here talks to hardware.
        freq: inner loop rate (Hz). Keep it well above the target update rate.
        gripper_max_speed: slew limit in gripper units per second. A policy
            setpoint updating at 10 Hz can jump the full range in one step,
            which slams the gripper; their stack limits this the same way.
            Set <= 0 to disable.
        target_frame: ``"tool0"`` (fingertips, what this repo's LeRobot EE
            datasets store) or ``"tcp"`` (the bspline/tidybot2 flange
            convention).
        q_init: initial virtual joint state, normally the arm's measured
            joints at startup. Defaults to zeros.
        gripper_init: initial gripper command.
    """

    def __init__(
        self,
        solver,
        send_joints: Callable[[np.ndarray, float], None] | None = None,
        freq: float = 200.0,
        gripper_max_speed: float = 3.0,
        target_frame: str = "tool0",
        q_init=None,
        gripper_init: float = 0.0,
        pos_reach_limit: float = 0.25,
        rot_reach_limit: float = 0.6,
    ) -> None:
        if target_frame not in TARGET_FRAMES:
            raise ValueError(f"target_frame must be one of {TARGET_FRAMES}, got {target_frame!r}")
        self.solver = solver
        self.send_joints = send_joints
        self.freq = float(freq)
        self.dt = 1.0 / self.freq
        self.gripper_max_speed = float(gripper_max_speed)
        self.target_frame = target_frame
        self.pos_reach_limit = float(pos_reach_limit)
        self.rot_reach_limit = float(rot_reach_limit)

        self._q = (np.zeros(6, dtype=np.float64) if q_init is None
                   else np.asarray(q_init, dtype=np.float64)[:6].copy())
        self._gripper = float(np.clip(gripper_init, 0.0, 1.0))

        self._lock = threading.Lock()
        self._target_pos: np.ndarray | None = None
        self._target_quat_wxyz: np.ndarray | None = None
        self._target_gripper = self._gripper

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # tool0 expressed in the link6 frame — constant by construction, read
        # off this solver's own model so it can never drift from the geometry
        # the IK actually solves against.
        self._T_link6_to_tool0 = self._measure_link6_to_tool0()
        # A pose given in the TCP convention becomes a tool0 pose via
        #   T_tool0 = T_tcp @ inv(T_LINK6_TO_TCP) @ T_link6_to_tool0
        self._T_tcp_to_tool0 = np.linalg.inv(T_LINK6_TO_TCP) @ self._T_link6_to_tool0

        mu = float(getattr(solver, "mu", 0.0))
        if mu > MU_DEPLOY_MAX:
            logger.warning(
                "EEFollower: solver mu=%.4g leaves a steady-state offset against a "
                "fixed target (~3.4 mm at the teleop default 0.02). Build the solver "
                "with mu <= %.3g for policy rollouts.", mu, MU_DEPLOY_MAX)

    def _measure_link6_to_tool0(self) -> np.ndarray:
        model, data = self.solver.model, self.solver.data
        link6 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        if link6 == -1:
            raise RuntimeError("link6 body missing from the solver's model")
        # Two configurations: the transform is rigid, so they must agree. This
        # is cheap insurance against a model change silently breaking the
        # frame conversion.
        mats = []
        for q in (np.zeros(6), np.array([0.3, -0.2, 0.5, 0.1, 0.4, -0.3])):
            self.solver.fk(q)
            A = np.eye(4)
            A[:3, 3] = data.xpos[link6]
            A[:3, :3] = data.xmat[link6].reshape(3, 3)
            B = np.eye(4)
            B[:3, 3] = data.site_xpos[self.solver.site_id]
            B[:3, :3] = data.site_xmat[self.solver.site_id].reshape(3, 3)
            mats.append(np.linalg.inv(A) @ B)
        if not np.allclose(mats[0], mats[1], atol=1e-9):
            raise RuntimeError("link6->tool0 is not rigid in this model")
        return mats[0]

    # ---------- target ----------

    def set_target(self, pos, quat_xyzw, gripper: float | None = None) -> None:
        """Replace the current target (latest wins — no queue, so a backlog
        can never make the arm chase stale commands)."""
        T = np.eye(4)
        T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
        T[:3, :3] = _mat_from_quat_xyzw(quat_xyzw)
        if self.target_frame == "tcp":
            T = T @ self._T_tcp_to_tool0
        with self._lock:
            self._target_pos = T[:3, 3].copy()
            self._target_quat_wxyz = _quat_wxyz_from_mat(T[:3, :3])
            if gripper is not None:
                self._target_gripper = float(np.clip(gripper, 0.0, 1.0))

    # ---------- stepping ----------

    def _clamp_reach(self, pos: np.ndarray, quat_wxyz: np.ndarray):
        """Bound the target to a ball around the arm's CURRENT pose.

        The same guard `ClutchPoseMapper` applies under teleop, and it is not
        optional here. Without it a wildly out-of-workspace command (a policy
        that extrapolates, a decode bug, a NaN-adjacent pose) stretches the arm
        to full extension, where it can end up pinned against the joint-2/3
        limits with the wrist at exactly ±pi/2 — full gimbal lock. That state
        is a genuine deadlock of the decoupled solver, not slow convergence:
        it does not recover even after minutes of stepping. Clamping turns any
        such command into a bounded pull in the right direction.

        Evaluated against the current pose every tick, so the arm still
        *travels* anywhere — it just cannot be yanked there in one step.
        """
        cur_pos, cur_quat = self.solver.fk(self._q)

        if self.pos_reach_limit > 0.0:
            delta = pos - cur_pos
            dist = float(np.linalg.norm(delta))
            if dist > self.pos_reach_limit:
                pos = cur_pos + delta * (self.pos_reach_limit / dist)

        if self.rot_reach_limit > 0.0:
            err = quat_to_rotvec(quat_mul(quat_wxyz, quat_conj(cur_quat)))
            angle = float(np.linalg.norm(err))
            if angle > self.rot_reach_limit:
                quat_wxyz = quat_mul(
                    rotvec_to_quat(err * (self.rot_reach_limit / angle)), cur_quat)

        return pos, quat_wxyz

    def step(self) -> tuple[np.ndarray, float]:
        """One control tick: a single IK step toward the target plus a
        slew-limited gripper update. Returns the commanded ``(qpos6, gripper)``.

        Safe to call before any target arrives — the arm simply holds."""
        with self._lock:
            pos, quat = self._target_pos, self._target_quat_wxyz
            target_gripper = self._target_gripper

        if pos is not None:
            pos, quat = self._clamp_reach(pos, quat)
            self._q = np.asarray(
                self.solver.solve(pos, quat, self._q), dtype=np.float64)[:6]

        if self.gripper_max_speed > 0.0:
            max_delta = self.gripper_max_speed * self.dt
            delta = np.clip(target_gripper - self._gripper, -max_delta, max_delta)
            self._gripper = float(np.clip(self._gripper + delta, 0.0, 1.0))
        else:
            self._gripper = target_gripper

        if self.send_joints is not None:
            self.send_joints(self._q.copy(), self._gripper)
        return self._q.copy(), self._gripper

    def _loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            self.step()
            next_tick += self.dt
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ee-follower", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---------- state ----------

    @property
    def qpos(self) -> np.ndarray:
        return self._q.copy()

    @property
    def gripper(self) -> float:
        return self._gripper

    def ee_pose(self, frame: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Current commanded EE pose as ``(pos, quat_xyzw)`` with ``w >= 0``,
        in ``frame`` (defaults to the follower's target frame). Use this to
        build the observation you feed the policy, so training and inference
        agree on the convention."""
        frame = frame or self.target_frame
        if frame not in TARGET_FRAMES:
            raise ValueError(f"frame must be one of {TARGET_FRAMES}, got {frame!r}")
        pos, quat_wxyz = self.solver.fk(self._q)
        T = np.eye(4)
        T[:3, 3] = pos
        flat = np.zeros(9)
        mujoco.mju_quat2Mat(flat, np.asarray(quat_wxyz, dtype=np.float64))
        T[:3, :3] = flat.reshape(3, 3)
        if frame == "tcp":
            T = T @ np.linalg.inv(self._T_tcp_to_tool0)
        q = _quat_wxyz_from_mat(T[:3, :3])
        xyzw = np.array([q[1], q[2], q[3], q[0]])
        if xyzw[3] < 0.0:
            xyzw = -xyzw
        return T[:3, 3].copy(), xyzw
