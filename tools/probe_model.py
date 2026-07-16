"""Probe the combined YAM arm + gripper model as built for the IK.

Dumps what `vr_teleop_kit.ik.model.build_model_with_tool0_site` produces —
the yam.xml arm merged with the linear_4310 gripper (i2rt mount transform
applied) plus the `tool0` / `j4_anchor` sites the IK pipeline needs — and
runs FK at a couple of poses so the chain can be eyeballed.

Run (model resolved like everywhere else: arg > YAM_XML > ./i2rt clone):
    python tools/probe_model.py [path/to/i2rt/robot_models/arm/yam/yam.xml]
"""

import sys

import mujoco
import numpy as np

from vr_teleop_kit.ik.model import DEFAULT_Q_REST, build_model_with_tool0_site

JOINT_TYPE_NAMES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}


def name_of(model: mujoco.MjModel, obj_type, idx: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, idx)
    return name if name is not None else f"<unnamed#{idx}>"


def print_model_summary(model: mujoco.MjModel) -> None:
    print("=== Model summary ===")
    print(f"  nq    (generalized coords) = {model.nq}")
    print(f"  nv    (DoFs)               = {model.nv}")
    print(f"  njnt  (joints)             = {model.njnt}")
    print(f"  nbody (bodies, incl world) = {model.nbody}")
    print(f"  nsite (sites)              = {model.nsite}")


def print_joints(model: mujoco.MjModel) -> None:
    print("\n=== Joints ===")
    for j in range(model.njnt):
        jname = name_of(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = JOINT_TYPE_NAMES.get(int(model.jnt_type[j]), f"type#{int(model.jnt_type[j])}")
        qadr = int(model.jnt_qposadr[j])
        rng = model.jnt_range[j]
        limited = bool(model.jnt_limited[j])
        rng_str = f"[{rng[0]:+.3f}, {rng[1]:+.3f}]" if limited else "(unlimited)"
        print(f"  [{j}] {jname:20s} type={jtype:6s} qpos_idx={qadr:>2}  range={rng_str}")


def print_bodies(model: mujoco.MjModel) -> None:
    print("\n=== Bodies ===")
    for b in range(model.nbody):
        bname = name_of(model, mujoco.mjtObj.mjOBJ_BODY, b)
        parent = int(model.body_parentid[b])
        pname = name_of(model, mujoco.mjtObj.mjOBJ_BODY, parent) if parent != b else "(self/world)"
        print(f"  [{b}] {bname:20s} parent={pname}")


def print_sites(model: mujoco.MjModel) -> None:
    if model.nsite == 0:
        return
    print("\n=== Sites ===")
    for s in range(model.nsite):
        sname = name_of(model, mujoco.mjtObj.mjOBJ_SITE, s)
        body = int(model.site_bodyid[s])
        bname = name_of(model, mujoco.mjtObj.mjOBJ_BODY, body)
        pos = model.site_pos[s]
        quat = model.site_quat[s]  # (w, x, y, z)
        print(
            f"  [{s}] {sname:20s} attached_to={bname:12s} "
            f"local_pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
            f"local_quat_wxyz=({quat[0]:+.4f},{quat[1]:+.4f},{quat[2]:+.4f},{quat[3]:+.4f})"
        )


def fk_dump(model: mujoco.MjModel, data: mujoco.MjData, label: str) -> None:
    print(f"\n=== FK at {label} ===")
    mujoco.mj_forward(model, data)

    bodies = ["world", "link1", "link2", "link3", "link4", "link5", "link6", "gripper"]
    for tname in bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tname)
        if bid == -1:
            continue
        pos = data.xpos[bid]
        q = data.xquat[bid]  # mujoco quaternion is (w, x, y, z)
        print(
            f"  body  '{tname:12s}' (id={bid:>2}): "
            f"pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
            f"quat_wxyz=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f})"
        )

    for sname in ["tool0", "j4_anchor"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
        if sid == -1:
            continue
        pos = data.site_xpos[sid]
        # Site world rotation is stored as a 3x3 row-major matrix in site_xmat.
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, data.site_xmat[sid])
        print(
            f"  site  '{sname:12s}' (id={sid:>2}): "
            f"pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
            f"quat_wxyz=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f})"
        )


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    model, data = build_model_with_tool0_site(model_path)

    print_model_summary(model)
    print_joints(model)
    print_bodies(model)
    print_sites(model)

    data.qpos[:] = 0
    fk_dump(model, data, "qpos = 0")

    data.qpos[:] = 0
    data.qpos[: len(DEFAULT_Q_REST)] = DEFAULT_Q_REST
    fk_dump(model, data, f"rest pose {np.round(DEFAULT_Q_REST, 3).tolist()}")


if __name__ == "__main__":
    main()
