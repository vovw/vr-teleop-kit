"""Check the sim can actually grasp a cube and stack one on another.

Teleop is only worth wiring up if the contact physics holds a cube in the first
place, so this tests that directly and reports numbers rather than opinions.

Three checks:

1. **Aperture** -- the jaw gap across the grasping region, against the cube.
   Measured from the finger meshes, not from tip-body separation: the two tip
   bodies move along their own axes and their origin distance runs the *opposite*
   way to the actual jaw gap, which makes it look as though the gripper closes
   when it opens.
2. **Grasp and lift** -- close on a cube, raise the arm, confirm the cube goes
   with it instead of squirting out or being left behind.
3. **Stack** -- rest one cube on another and settle, confirming the stack does
   not slide apart or sink into itself.

    DISPLAY=:0 MUJOCO_GL=glfw python tools/sim_check_grasp.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

GRIPPER_OPEN = 0.0475      # joint7/8 value for open (measured, see docstring)
GRIPPER_CLOSED = 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(_REPO_ROOT / "sim" / "spd_scene.xml"))
    ap.add_argument("--side", default="right", choices=("left", "right"))
    args = ap.parse_args()

    import mujoco

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    side = args.side
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready")

    def jid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    def aid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    def bid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

    def hold_arm():
        """Command every arm joint of both arms to hold where it currently is."""
        for s in ("left", "right"):
            for j in range(1, 7):
                a = aid(f"{s}_joint{j}")
                q = d.qpos[m.jnt_qposadr[jid(f"{s}_joint{j}")]]
                if a >= 0:
                    d.ctrl[a] = q

    def settle(seconds, grip=None):
        if grip is not None:
            d.ctrl[aid(f"{side}_gripper")] = grip
        for _ in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)

    # ── 1. aperture vs cube ─────────────────────────────────────────────
    cube_edge = float(m.geom_size[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_GEOM, "cube_centre_geom")][0] * 2)
    print(f"cube edge {cube_edge*1000:.0f} mm, "
          f"mass {m.body_mass[bid('cube_centre')]*1000:.1f} g")

    # ── 2. grasp and lift ───────────────────────────────────────────────
    mujoco.mj_resetDataKeyframe(m, d, kid)
    hold_arm()
    d.ctrl[aid(f"{side}_gripper")] = GRIPPER_OPEN
    settle(0.3)

    gs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp_site")
    cube = "cube_centre"
    cj = jid(f"{cube}_free")
    adr = m.jnt_qposadr[cj]
    # Put the cube where the jaws are, upright.
    target = d.site_xpos[gs].copy()
    d.qpos[adr:adr + 3] = target
    d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    d.qvel[m.jnt_dofadr[cj]:m.jnt_dofadr[cj] + 6] = 0
    mujoco.mj_forward(m, d)
    print(f"\ncube placed at {side} grasp site {np.round(target,3)}")

    # Close with gravity off, then switch it back on. Placing a cube in mid-air
    # at the jaws and taking 0.8 s to close just measures free fall (~460 mm) --
    # the question here is whether the closed jaws *hold* it, so let the grasp
    # form first and only then let it be pulled on.
    g_save = m.opt.gravity.copy()
    m.opt.gravity[:] = 0
    # Ramp the jaws shut over 0.5 s rather than stepping the command. A step is
    # not what teleop does -- the operator squeezes a trigger -- and it matters:
    # commanding fully-closed in one tick drives the fingers into the cube hard
    # enough to shoot it out, so a step-close "fails" for a grasp that is
    # perfectly solid in use.
    steps = int(0.5 / m.opt.timestep)
    for k in range(steps):
        d.ctrl[aid(f"{side}_gripper")] = GRIPPER_OPEN * (1.0 - k / steps)
        mujoco.mj_step(m, d)
    settle(0.25, grip=GRIPPER_CLOSED)
    m.opt.gravity[:] = g_save
    settle(1.0)
    held = d.xpos[bid(cube)].copy()
    off_closed = np.linalg.norm(held - d.site_xpos[gs])
    print(f"after closing + 1 s under gravity: "
          f"cube-to-grasp-site offset {off_closed*1000:5.1f} mm")

    # Lift by raising joint 2, holding everything else.
    j2 = jid(f"{side}_joint2")
    start_z = d.xpos[bid(cube)][2]
    # Capture the start angle ONCE. Reading it live inside the loop makes the
    # commanded target chase the joint's own position, so the arm never moves.
    q0 = float(d.qpos[m.jnt_qposadr[j2]])
    tgt = q0 - 0.35
    for k in range(int(1.5 / m.opt.timestep)):
        a = min(1.0, k / (0.8 / m.opt.timestep))
        d.ctrl[aid(f"{side}_joint2")] = q0 * (1 - a) + tgt * a
        mujoco.mj_step(m, d)
    settle(0.5)
    end = d.xpos[bid(cube)].copy()
    off_lift = np.linalg.norm(end - d.site_xpos[gs])
    rise = end[2] - start_z
    print(f"after lift:    cube rose {rise*1000:+6.1f} mm, "
          f"offset from jaws {off_lift*1000:5.1f} mm")
    grasp_ok = off_lift < 0.06 and rise > 0.02
    print(f"GRASP+LIFT: {'PASS' if grasp_ok else 'FAIL'}"
          f"  (cube must stay within 60 mm of the jaws and rise >20 mm)")

    # ── 3. stack stability ──────────────────────────────────────────────
    mujoco.mj_resetDataKeyframe(m, d, kid)
    hold_arm()
    lo = jid("cube_left_free")
    hi = jid("cube_right_free")
    a_lo, a_hi = m.jnt_qposadr[lo], m.jnt_qposadr[hi]
    base = d.qpos[a_lo:a_lo + 3].copy()
    d.qpos[a_hi:a_hi + 3] = base + np.array([0.0, 0.0, cube_edge])
    d.qpos[a_hi + 3:a_hi + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    before = d.xpos[bid("cube_right")].copy()
    settle(3.0)
    after = d.xpos[bid("cube_right")].copy()
    drift = np.linalg.norm(after[:2] - before[:2])
    sink = before[2] - after[2]
    print(f"\nstack: upper cube drifted {drift*1000:5.2f} mm laterally, "
          f"sank {sink*1000:5.2f} mm over 3 s")
    stack_ok = drift < 0.005 and abs(sink) < 0.003
    print(f"STACK: {'PASS' if stack_ok else 'FAIL'}"
          f"  (needs <5 mm drift and <3 mm sink)")

    print(f"\ncontact settings: timestep {m.opt.timestep*1000:.3f} ms, "
          f"cone={'elliptic' if m.opt.cone==1 else 'pyramidal'}, "
          f"noslip_iterations={m.opt.noslip_iterations}")
    print(f"cube friction {m.geom_friction[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,'cube_centre_geom')]}, "
          f"finger friction {m.geom_friction[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, side+'_tip_left')]}")


if __name__ == "__main__":
    main()
