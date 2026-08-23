"""Script a bimanual cube-stacking demo and record it as a LeRobot episode.

Produces a reference episode without a headset: the left arm puts its cube on
the centre cube, then the right arm puts its cube on top, giving a stack of
three. Written in the same dataset schema as examples/record_sim_cubes.py and
examples/record_bi_yam.py, so it sits alongside teleop episodes.

Motion comes from damped-least-squares IK on the full scene (mj_jacBody on the
gripper body), not from the repo's DecoupledIKSolver -- that builds its own
single-arm model around a `tool0` site, which this scene does not have.

Two things the geometry forces:

* **Which arm takes which cube.** The reachable envelope at cube height with the
  jaws vertical is small and shrinks fast with row distance. Each arm can reach
  its own cube and the centre, but not across the table, so the left arm handles
  the left cube and the right arm the right one.
* **Yaw is left free.** Constraining it wastes a DOF for no benefit -- a cube
  does not care how the jaws are rotated about the approach axis.

The jaws close along the gripper's own +Y, which is world Y here, i.e. across
the row rather than along it -- so closing on one cube does not swipe its
neighbour.

    DISPLAY=:0 MUJOCO_GL=glfw python tools/sim_demo_stack.py \\
        --repo-id atharva/sim-cubes-demo --root data/sim-cubes-demo
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from vr_teleop_kit.log import get_logger, setup_logging  # noqa: E402
from vr_teleop_kit.relay.sim_frames import CAM_IDS, SimFrameBus  # noqa: E402

ARM_DOFS = 6
HANDS = ("left", "right")
GRIPPER_OPEN_QPOS = 0.0475
GRASP_OFFSET = 0.1347          # grasp_site offset down the gripper body's -Z
APPROACH_DOWN = np.array([0.0, 0.0, -1.0])


class Rig:
    def __init__(self, scene: str, fps: int, log) -> None:
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.log = log
        self.fps = fps
        self.substeps = max(1, round((1.0 / fps) / self.model.opt.timestep))
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "start")
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)

        self.jadr = {f"{s}_joint{j}": self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_joint{j}")]
            for s in HANDS for j in range(1, 9)}
        self.dofadr = {f"{s}_joint{j}": self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_joint{j}")]
            for s in HANDS for j in range(1, 7)}
        self.act = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                    for n in [f"{s}_joint{j}" for s in HANDS for j in range(1, 7)]
                    + [f"{s}_gripper" for s in HANDS]}
        self.body = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                     for n in ("left_gripper", "right_gripper",
                               "cube_left", "cube_centre", "cube_right")}
        self.limits = {s: np.array([self.model.jnt_range[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_joint{j}")]
            for j in range(1, 7)]) for s in HANDS}
        self.hold_current()
        mujoco.mj_forward(self.model, self.data)
        self.cams = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, c)
                     for c in CAM_IDS]
        self.renderer = mujoco.Renderer(self.model, 480, 640)

    # ── state helpers ───────────────────────────────────────────────────
    def q(self, side): return np.array([self.data.qpos[self.jadr[f"{side}_joint{j}"]]
                                        for j in range(1, 7)])

    def grip(self, side): return float(self.data.qpos[self.jadr[f"{side}_joint7"]])

    def hold_current(self) -> None:
        for s in HANDS:
            for j in range(1, 7):
                self.data.ctrl[self.act[f"{s}_joint{j}"]] = \
                    self.data.qpos[self.jadr[f"{s}_joint{j}"]]
            self.data.ctrl[self.act[f"{s}_gripper"]] = self.grip(s)

    def cube_pos(self, name): return self.data.xpos[self.body[name]].copy()

    # ── IK ──────────────────────────────────────────────────────────────
    def ik(self, side, target_pos, approach=APPROACH_DOWN, iters=400):
        """Position + approach-axis IK (yaw free). Returns (q, pos_err, axis_err)."""
        mj = self.mj
        body = self.body[f"{side}_gripper"]
        A = [self.jadr[f"{side}_joint{j}"] for j in range(1, 7)]
        D = [self.dofadr[f"{side}_joint{j}"] for j in range(1, 7)]
        L = self.limits[side]
        tmp = mj.MjData(self.model)
        q = self.q(side).copy()
        jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))
        ep = er = 1.0
        for _ in range(iters):
            tmp.qpos[:] = self.data.qpos
            for k, a in enumerate(A):
                tmp.qpos[a] = q[k]
            mj.mj_kinematics(self.model, tmp); mj.mj_comPos(self.model, tmp)
            p = tmp.xpos[body]; R = tmp.xmat[body].reshape(3, 3)
            e_p = target_pos - p
            e_r = np.cross(-R[:, 2], approach)
            ep, er = np.linalg.norm(e_p), np.linalg.norm(e_r)
            if ep < 3e-4 and er < 5e-3:
                break
            mj.mj_jacBody(self.model, tmp, jacp, jacr, body)
            J = np.vstack([jacp[:, D], jacr[:, D]])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.06 ** 2 * np.eye(6),
                                       np.concatenate([e_p, 0.8 * e_r]))
            q = np.clip(q + np.clip(dq, -0.12, 0.12), L[:, 0], L[:, 1])
        return q, ep, er

    # ── motion ──────────────────────────────────────────────────────────
    def drive(self, targets: dict[str, np.ndarray], grips: dict[str, float],
              seconds: float, on_tick) -> None:
        """Interpolate joint + gripper commands over `seconds`, stepping physics.

        Commands are ramped rather than stepped: the arms are position
        controlled at the hardware's own modest gains, and a step command both
        overshoots and (for the gripper) flings a cube out of the jaws.
        """
        ticks = max(1, int(seconds * self.fps))
        q0 = {s: self.q(s) for s in HANDS}
        g0 = {s: self.grip(s) for s in HANDS}
        for t in range(1, ticks + 1):
            a = t / ticks
            for s in HANDS:
                tgt = targets.get(s)
                qd = q0[s] + a * (tgt - q0[s]) if tgt is not None else q0[s]
                for j in range(1, 7):
                    self.data.ctrl[self.act[f"{s}_joint{j}"]] = qd[j - 1]
                gd = grips.get(s)
                if gd is not None:
                    self.data.ctrl[self.act[f"{s}_gripper"]] = g0[s] + a * (gd - g0[s])
            for _ in range(self.substeps):
                self.mj.mj_step(self.model, self.data)
            on_tick()

    def render(self):
        out = {}
        for cid, cam in zip(CAM_IDS, self.cams):
            self.renderer.update_scene(self.data, camera=cam)
            out[cid] = self.renderer.render()
        return out

    # ── observation / action vectors (same layout as the recorders) ──────
    def obs_vector(self):
        v = []
        for s in HANDS:
            v += [float(self.data.qpos[self.jadr[f"{s}_joint{j}"]]) for j in range(1, 7)]
            v.append(1.0 - float(np.clip(self.grip(s) / GRIPPER_OPEN_QPOS, 0, 1)))
        return np.asarray(v, dtype=np.float32)

    def action_vector(self):
        v = []
        for s in HANDS:
            v += [float(self.data.ctrl[self.act[f"{s}_joint{j}"]]) for j in range(1, 7)]
            g = float(self.data.ctrl[self.act[f"{s}_gripper"]])
            v.append(1.0 - float(np.clip(g / GRIPPER_OPEN_QPOS, 0, 1)))
        return np.asarray(v, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(_REPO_ROOT / "sim" / "spd_scene.xml"))
    ap.add_argument("--repo-id", default="atharva/sim-cubes-demo")
    ap.add_argument("--root", default=str(_REPO_ROOT / "data" / "sim-cubes-demo"))
    ap.add_argument("--task", default="stack the three cubes")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-record", action="store_true",
                    help="run the motion and report, without writing a dataset")
    ap.add_argument("--publish", action="store_true",
                    help="also publish frames to the shared-memory bus, so the "
                         "demo can be watched live in the headset or viser")
    args = ap.parse_args()

    setup_logging(level=logging.INFO)
    log = get_logger(__name__, "demo")
    rig = Rig(args.scene, args.fps, log)
    E = float(rig.model.geom_size[
        rig.mj.mj_name2id(rig.model, rig.mj.mjtObj.mjOBJ_GEOM, "cube_centre_geom")][0] * 2)

    bus = SimFrameBus(create=True) if args.publish else None
    dataset = None
    if not args.no_record:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        root = Path(args.root)
        if root.exists():
            if args.overwrite:
                shutil.rmtree(root)
            else:
                raise SystemExit(f"{root} exists — pass --overwrite")
        names = [f"{h}_{n}" for h in HANDS
                 for n in [f"joint_{j+1}" for j in range(ARM_DOFS)] + ["gripper"]]
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id, fps=args.fps, root=args.root,
            robot_type="bi_yam_sim", use_videos=True, image_writer_threads=12,
            features={
                "observation.state": {"dtype": "float32", "shape": (len(names),),
                                      "names": names},
                "action": {"dtype": "float32", "shape": (len(names),), "names": names},
                **{f"observation.images.{c}": {"dtype": "video",
                                               "shape": (480, 640, 3),
                                               "names": ["height", "width", "channels"]}
                   for c in CAM_IDS},
            })
        log.info("recording to %s", dataset.root)

    frames = {"n": 0}

    def on_tick():
        imgs = rig.render()
        if bus is not None:
            bus.write_state(rig.data.qpos)
            for cid, rgb in imgs.items():
                bus.write(cid, rgb[:, :, ::-1])
        if dataset is not None:
            f = {"observation.state": rig.obs_vector(),
                 "action": rig.action_vector(), "task": args.task}
            for cid, rgb in imgs.items():
                f[f"observation.images.{cid}"] = rgb
            dataset.add_frame(f)
        frames["n"] += 1

    def goto(side, pos, seconds, grip=None, label=""):
        q, ep, er = rig.ik(side, pos)
        if ep > 0.01:
            log.warning("%s %s: IK short by %.1f mm — motion may miss",
                        side, label, ep * 1000)
        rig.drive({side: q}, {side: grip} if grip is not None else {}, seconds, on_tick)

    def pick_place(side, cube, place_xy, place_z, label):
        """Approach above, descend, close, lift, traverse, descend, release."""
        c = rig.cube_pos(cube)
        above = np.array([c[0], c[1], 0.16 + GRASP_OFFSET])
        at = np.array([c[0], c[1], c[2] + GRASP_OFFSET])
        log.info("%s: %s -> approach", side, label)
        goto(side, above, 2.0, grip=GRIPPER_OPEN_QPOS, label="above pick")
        goto(side, at, 1.5, label="at pick")
        log.info("%s: closing", side)
        rig.drive({}, {side: 0.0}, 1.0, on_tick)          # ramped close
        log.info("%s: lifting", side)
        goto(side, np.array([c[0], c[1], 0.20 + GRASP_OFFSET]), 1.5, label="lift")
        tgt_above = np.array([place_xy[0], place_xy[1], 0.20 + GRASP_OFFSET])
        goto(side, tgt_above, 2.0, label="above place")
        goto(side, np.array([place_xy[0], place_xy[1], place_z + GRASP_OFFSET]),
             2.0, label="at place")
        log.info("%s: releasing", side)
        rig.drive({}, {side: GRIPPER_OPEN_QPOS}, 0.8, on_tick)
        goto(side, tgt_above, 1.2, label="retreat")

    t0 = time.perf_counter()
    centre = rig.cube_pos("cube_centre")
    place_xy = (float(centre[0]), float(centre[1]))

    pick_place("left", "cube_left", place_xy, E * 1.5, "left cube onto centre")
    # Get the left arm out of the way before the right arm comes in over the stack.
    q_park, _, _ = rig.ik("left", np.array([-0.30, -0.55, 0.30 + GRASP_OFFSET]))
    rig.drive({"left": q_park}, {}, 2.0, on_tick)
    pick_place("right", "cube_right", place_xy, E * 2.5, "right cube onto stack")

    rig.drive({}, {}, 1.5, on_tick)          # settle

    # ── did it work? ────────────────────────────────────────────────────
    pos = {n: rig.cube_pos(n) for n in ("cube_left", "cube_centre", "cube_right")}
    log.info("final cube positions:")
    for n, p in pos.items():
        log.info("  %-12s %s", n, np.round(p, 4))
    zs = sorted((p[2], n) for n, p in pos.items())
    heights = [z for z, _ in zs]
    xy_spread = max(np.linalg.norm(pos[a][:2] - pos[b][:2])
                    for a in pos for b in pos)
    stacked = (abs(heights[0] - E / 2) < 0.006
               and abs(heights[1] - E * 1.5) < 0.012
               and abs(heights[2] - E * 2.5) < 0.018
               and xy_spread < 0.030)
    log.info("heights %s (want %.4f/%.4f/%.4f), max xy spread %.1f mm",
             np.round(heights, 4), E / 2, E * 1.5, E * 2.5, xy_spread * 1000)
    log.info("STACK OF THREE: %s", "YES" if stacked else "NO")
    log.info("%d frames, %.1f s of episode, %.1f s wall",
             frames["n"], frames["n"] / args.fps, time.perf_counter() - t0)

    if dataset is not None:
        if frames["n"]:
            dataset.save_episode()
        if hasattr(dataset, "finalize"):
            dataset.finalize()
        log.info("saved %d episode(s) at %s", dataset.num_episodes, dataset.root)
    if bus is not None:
        bus.close()


if __name__ == "__main__":
    main()
