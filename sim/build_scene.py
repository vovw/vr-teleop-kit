"""Generate a MuJoCo scene that pixel-matches the real bimanual YAM rig.

Everything measurable about this rig is measured, not guessed:

* **Camera intrinsics** come from ``sim/camera_intrinsics.json``, read out of
  each D405's calibration EEPROM by ``tools/sim_dump_intrinsics.py``. They are
  written into MuJoCo's *intrinsic* camera form (``sensorsize`` / ``focal`` /
  ``principal`` / ``resolution``) rather than a single ``fovy``, so fx, fy and
  the off-centre principal point are all reproduced exactly. ``fovy`` alone
  cannot express fx != fy or a shifted principal point.
* **Table** is 1.60 x 1.60 m (measured), top surface at ``z = 0``, which is the
  origin of the whole scene.
* **Top camera** sits 0.90 m above that surface, between the arm bases. Its
  remaining pose (forward offset, pitch, roll, yaw) is *fitted* to the real
  image by ``tools/sim_fit_top_camera.py`` and stored in ``sim/top_camera_pose.json``;
  until that file exists we fall back to a geometric first guess.

World frame: origin at the centre of the tabletop, +X right (as the top camera
sees it), +Y away from the camera toward the far edge, +Z up.

MuJoCo's own pixel geometry note: with the intrinsic form, MuJoCo's ``principal``
is an offset from the image *centre* in length units, and its camera frame has
+Y pointing up while image rows count downward -- hence the sign flip on the
vertical component below.

    python sim/build_scene.py                       # -> sim/spd_scene.xml
    python sim/build_scene.py --table-color neutral
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

# ── Measured rig geometry ────────────────────────────────────────────────
TABLE_SIZE = 1.60          # m, square tabletop (measured)
TABLE_THICKNESS = 0.025    # m, board thickness
TABLE_HEIGHT = 0.75        # m, tabletop above floor (assumed; see README note)
TOP_CAM_HEIGHT = 0.90      # m, top camera above the tabletop (measured)
ARM_BASE_SEPARATION = 0.50 # m, base-to-base across the near edge (measured)
ARM_BASE_Y = -TABLE_SIZE / 2   # bases sit at the near edge
# Yaw of the arm bases about Z. +90 deg is empirically what puts the grippers in
# the top camera's view at the joint angles actually recorded in calib/ (a sweep
# over yaw peaks at 75-105 deg). Do NOT re-derive this from the rest pose: at
# j2=0 -- a joint limit -- the arm reaches along -X and the answer looks like
# -90 deg, which is ~180 deg wrong for the j2 ~ 1.0-2.2 configurations real
# teleop actually uses, and puts the grippers behind the camera.
ARM_MOUNT_YAW_DEG = 90.0

# Table colour. "asshot" is the median tabletop colour measured off
# greenscreen/top.png, where the D405's auto-white-balance rendered this white
# melamine strongly cyan. "neutral" is the same surface grey-world corrected,
# i.e. roughly its true albedo.
#
# Caveat worth knowing: the cyan is NOT a property of the table. The later
# calib/ recording of the same tabletop came out near-neutral (B-R of -7 against
# +64 in the greenscreen frame), so "asshot" only matches frames shot under that
# same white balance. "neutral" is the more portable choice.
TABLE_RGBA_ASSHOT = (112 / 255, 179 / 255, 180 / 255, 1.0)
TABLE_RGBA_NEUTRAL = (0.616, 0.616, 0.616, 1.0)

# Calibrated light multiplier, from tools/sim_calibrate_lighting.py: makes a
# flat patch of rendered tabletop match the brightness measured in the real
# frame (to +0.2%). Calibrated against the "asshot" material.
LIGHT_SCALE = 0.9012

# An arbitrary but self-consistent pixel pitch. MuJoCo only cares about the
# ratios focal/sensorsize and principal/sensorsize, so the absolute value is
# free; 6 um is about right for a D405 colour pixel in its binned 640x480 mode.
PIXEL_PITCH = 6.0e-6

# grasp_site sits this far down the gripper body's -Z (from the i2rt XML).
GRASP_SITE_Z = 0.1347

# ── Real arm control parameters ──────────────────────────────────────────
# Verbatim from i2rt/i2rt/robots/config/yam.yml, which is what drives the
# hardware. Joints 1-3 are DM4340s and 4-6 are DM4310s, hence the split gains;
# a single uniform kp (this file used to use 120 everywhere) makes the wrist
# eight to twelve times too stiff and the shoulder too soft.
ARM_KP = (80.0, 80.0, 80.0, 10.0, 10.0, 10.0)
ARM_KD = (5.0, 5.0, 5.0, 1.5, 1.5, 1.5)
# Coulomb friction per joint (Nm). Real and non-negligible: without it a
# commanded pose drifts under its own dynamics in a way the hardware does not.
ARM_FRICTIONLOSS = (0.3, 0.3, 0.3, 0.06, 0.06, 0.06)
# The hardware feeds gravity torques forward scaled per joint, slightly
# OVER-compensating 2, 3 and 4. MuJoCo's gravcomp is per body, not per joint,
# so those factors cannot be reproduced exactly; bodies get full compensation
# and joints 2-4 are therefore marginally less assisted here than on the robot.
ARM_GRAVITY_COMP_FACTOR = (1.0, 1.1, 1.1, 1.2, 1.0, 1.0)

# Where real demonstrations actually begin, per arm: the median first frame of
# the training episodes, from molmoact2/examples/yam/configs/yam_{left,right}.yaml.
# The all-zero pose is the *mechanical rest* pose the arms park at on shutdown,
# not where teleop starts, so a sim episode opening at all-zeros starts from a
# configuration no real episode does. Gripper is held fully open for a reset.
REAL_START_JOINTS = {
    "left":  (0.021, 0.015, 0.049, -0.133, 0.174, 0.234),
    "right": (0.015, 0.016, 0.130, -0.418, -0.029, -0.023),
}

# ── Task objects: three 3D-printed cubes ────────────────────────────────
# 35 mm cubes, in a row 0.44 m out from the arm bases, with a 0.10 m gap
# between neighbours (so centres are 0.035 + 0.10 = 0.135 m apart).
#
# 0.44 m is the CENTRE of the spawn region, chosen so the region's near edge
# lands at 0.40 m (0.44 - 0.08/2). Blocks therefore spawn between 0.40 m and
# 0.48 m out. Note this is deliberately further than the 0.35 m that the
# conditioning table below argues for -- see the warning under SPAWN_REGION_M.
#
# Originally chosen from the arm's *conditioning*, not just whether IK converges.
# The stack point is the centre position, and reaching x=0 gets bad fast with
# distance -- measured at block height, and one block up:
#
#   dist   centre, table level        centre, +1 block
#   0.30   cond  13, margin 0.56      cond  14, margin 0.43
#   0.35   cond  16, margin 0.26      cond  18, margin 0.34
#   0.40   cond  22, margin 0.00      cond  27, margin 0.17
#   0.45   cond 196, margin 0.00      out of reach
#
# "margin" is radians to the nearest joint limit. At 0.40-0.45 m joints 4 and 6
# sit exactly on their limits with a near-singular Jacobian, so the teleop's
# adaptive damping ramps up and the reach limiter absorbs the operator's motion:
# the arm goes stiff and stops following the hand, which reads as broken IK
# rather than as a workspace edge. Placing the third block was outright
# unreachable at 0.45 m, which is why that step felt impossible.
CUBE_EDGE = 0.035
CUBE_GAP = 0.10
CUBE_DIST_FROM_ARMS = 0.44

# Mass from the print spec rather than a guess. A slicer's "15% infill" applies
# only to the interior -- the part still has solid perimeters and solid top and
# bottom layers, which for a small cube is most of the plastic. Ignoring the
# shell would under-estimate the mass by about half (12 g against 23 g), and a
# cube that light skitters away instead of being pushed.
PLA_DENSITY = 1240.0        # kg/m^3, solid PLA
CUBE_INFILL = 0.15          # slicer infill fraction
CUBE_SHELL_MM = 1.2         # ~3 perimeters at a 0.4 mm nozzle


def _cube_mass() -> float:
    outer = CUBE_EDGE ** 3
    inner_edge = max(CUBE_EDGE - 2 * (CUBE_SHELL_MM / 1000.0), 0.0)
    inner = inner_edge ** 3
    solid_volume = (outer - inner) + inner * CUBE_INFILL
    return solid_volume * PLA_DENSITY


# left -> right as the top camera sees it (-X to +X).
CUBE_COLOURS = {
    "cube_left":   (0.05, 0.05, 0.05, 1.0),   # black
    "cube_centre": (0.70, 0.06, 0.06, 1.0),   # red
    "cube_right":  (0.70, 0.06, 0.06, 1.0),   # red
}
CUBE_ORDER = ("cube_left", "cube_centre", "cube_right")


def cube_poses() -> list[tuple[str, float, float, float]]:
    """(name, x, y, z) for each cube, resting on the tabletop."""
    pitch = CUBE_EDGE + CUBE_GAP
    y = ARM_BASE_Y + CUBE_DIST_FROM_ARMS
    z = CUBE_EDGE / 2.0
    return [(n, (i - 1) * pitch, y, z) for i, n in enumerate(CUBE_ORDER)]

# The "ready" keyframe is taken from a real recorded teleop pose rather than
# hand-picked joint values. A hand-picked pose silently goes stale the moment
# the base yaw changes -- the previous one was tuned for a -90 deg mount and put
# the arms out of frame once that was corrected to +90. calib/pose_04 has both
# grippers comfortably inside the top camera's view.
READY_POSE_DIR = "calib/pose_04"
# Jaw travel: joint7/joint8 at 0 is closed, 0.0475 fully open (~95 mm
# between the finger faces). Measured, not inferred -- the two tip
# *bodies* travel along their own axes and their origin separation runs
# the opposite way, which reads as closing when the jaws open.
GRIPPER_OPEN_QPOS = 0.0475
# Fallback if calib/ has not been captured yet.
READY_ARM_QPOS_FALLBACK = (0.0, 1.4, 1.2, 0.0, 0.3, 0.0, 0.02, 0.02)


def _ready_arm_joints() -> dict[str, float]:
    """Ready-pose value for each arm joint, keyed by joint name.

    Recorded gripper state is 0=open..1=closed (record_bi_yam.py inverts i2rt's
    convention); the sim's coupled slide joints run 0=closed..0.0475=open.
    """
    src = _REPO_ROOT / READY_POSE_DIR / "joints.json"
    out: dict[str, float] = {}
    if not src.exists():
        for side in ("left", "right"):
            for j, v in enumerate(READY_ARM_QPOS_FALLBACK[:6], start=1):
                out[f"{side}_joint{j}"] = v
            out[f"{side}_joint7"] = out[f"{side}_joint8"] = 0.02
        return out
    rec = json.loads(src.read_text())
    for side in ("left", "right"):
        for j, v in enumerate(rec[f"{side}_joints"], start=1):
            out[f"{side}_joint{j}"] = float(v)
        grip = (1.0 - float(rec[f"{side}_gripper"])) * 0.0475
        out[f"{side}_joint7"] = out[f"{side}_joint8"] = grip
    return out


def _add_keyframes(mj: ET.Element) -> None:
    """Append the 'start' and 'ready' keyframes, laid out to match compiled qpos.

    Two poses, because they mean different things:

    * **start** -- where real demonstrations actually begin (REAL_START_JOINTS,
      the median first frame of the training episodes). This is what a sim
      episode should open from. It is close to, but not, the all-zero pose:
      all-zeros is the *mechanical rest* pose the arms park at on shutdown.
    * **ready** -- a real recorded teleop pose (calib/pose_04) with both arms up
      over the table. Useful for looking at the scene, since at the start pose
      the arms are folded back near the near edge and largely out of the top
      camera's view -- exactly as on the robot.

    A keyframe's qpos is positional and MuJoCo orders it by joint declaration
    order, so hand-assembling the vector breaks silently whenever a body is
    added or moved. The cubes are declared before the arms, so their freejoint
    DOFs come first; an arms-then-cubes vector quietly loaded arm joint angles
    into cube positions. So: compile the model as built, then fill by joint name.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(ET.tostring(mj, encoding="unicode"))

    def qpos_for(arm_joints: dict[str, float]) -> list[float]:
        q = list(model.qpos0)
        for name, val in arm_joints.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                q[model.jnt_qposadr[jid]] = val
        for name, cx, cy, cz in cube_poses():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
            if jid >= 0:
                a = model.jnt_qposadr[jid]
                q[a:a + 7] = [cx, cy, cz, 1.0, 0.0, 0.0, 0.0]
        return q

    start: dict[str, float] = {}
    for side, vals in REAL_START_JOINTS.items():
        for j, v in enumerate(vals, start=1):
            start[f"{side}_joint{j}"] = float(v)
        # Reset holds the gripper fully open, as the real resets do.
        start[f"{side}_joint7"] = start[f"{side}_joint8"] = GRIPPER_OPEN_QPOS

    kf = ET.SubElement(mj, "keyframe")
    ET.SubElement(kf, "key", name="start",
                  qpos=" ".join(f"{v:.6g}" for v in qpos_for(start)))
    ET.SubElement(kf, "key", name="ready",
                  qpos=" ".join(f"{v:.6g}" for v in qpos_for(_ready_arm_joints())))


_PREFIXED_TAGS = ("body", "joint", "geom", "site", "camera", "light")


def _prefix_tree(elem: ET.Element, prefix: str) -> None:
    """Namespace every name in an arm subtree so two copies can coexist."""
    for node in elem.iter():
        if node.tag in _PREFIXED_TAGS and node.get("name"):
            node.set("name", prefix + node.get("name"))


def _prefix_refs(root: ET.Element, prefix: str) -> None:
    """Rewrite the cross-references that go with the renamed elements."""
    for eq in root.iter("joint"):
        for attr in ("joint1", "joint2"):
            if eq.get(attr):
                eq.set(attr, prefix + eq.get(attr))
    for ex in root.iter("exclude"):
        for attr in ("body1", "body2"):
            if ex.get(attr):
                ex.set(attr, prefix + ex.get(attr))


def _arm_source() -> ET.Element:
    """The combined YAM + linear_4310 model, as i2rt itself builds it.

    ``record_bi_yam.py`` calls ``get_yam_robot`` without a ``gripper_type``,
    so the recorded robot is YAM + LINEAR_4310; we ask i2rt for exactly that
    pairing rather than hand-merging the two XMLs (it also applies the
    per-arm mounting transform from the gripper's YAML config).
    """
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    return ET.parse(path).getroot()


def _look_at_quat(cam_pos, target, up) -> tuple[float, float, float, float]:
    """Quaternion (w,x,y,z) aiming a MuJoCo camera at `target`.

    MuJoCo cameras look down their own -Z with +Y up, so the rotation's third
    column is the *negated* view direction.
    """
    import numpy as _np

    fwd = _np.asarray(target, float) - _np.asarray(cam_pos, float)
    n = _np.linalg.norm(fwd)
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    z = -fwd / n                                   # camera +Z
    u = _np.asarray(up, float)
    y = u - _np.dot(u, z) * z                      # camera +Y, orthogonalised
    if _np.linalg.norm(y) < 1e-6:                  # up parallel to view
        u = _np.array([1.0, 0.0, 0.0])
        y = u - _np.dot(u, z) * z
    y /= _np.linalg.norm(y)
    x = _np.cross(y, z)                            # right-handed: x = y X z
    R = _np.stack([x, y, z], axis=1)
    w = _np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-8:
        return (w, (R[2, 1] - R[1, 2]) / (4 * w),
                (R[0, 2] - R[2, 0]) / (4 * w),
                (R[1, 0] - R[0, 1]) / (4 * w))
    return (1.0, 0.0, 0.0, 0.0)


def _camera_intrinsic_attrs(intr: dict) -> dict[str, str]:
    """Turn measured pixel intrinsics into MuJoCo's intrinsic camera form."""
    w, h = intr["width"], intr["height"]
    p = PIXEL_PITCH
    # Principal point as an offset from the image centre. Both signs here are
    # empirically pinned by tools/sim_verify_intrinsics.py, because MuJoCo's
    # two axes do not agree with each other: a positive `principal` x moves the
    # principal point toward *smaller* u, while a positive y moves it toward
    # smaller v. Getting either wrong shifts the whole render by 2*offset,
    # which is ~10 px on this rig -- easily mistaken for a bad camera pose.
    dx = -(intr["ppx"] - w / 2.0) * p
    dy = -(intr["ppy"] - h / 2.0) * p
    return {
        "resolution": f"{w} {h}",
        "sensorsize": f"{w * p:.10g} {h * p:.10g}",
        "focal": f"{intr['fx'] * p:.10g} {intr['fy'] * p:.10g}",
        "principal": f"{dx:.10g} {dy:.10g}",
    }


def _default_top_pose() -> dict:
    """Geometric first guess, used only until the fit has been run.

    Aims the camera at the middle of the table from its measured height, which
    is close enough for the optimiser in ``tools/sim_fit_top_camera.py`` to
    converge from.
    """
    y_cam = ARM_BASE_Y            # between the bases, at the near edge
    aim_y = 0.0                   # centre of the table
    # A MuJoCo camera at zero euler looks straight down (-Z). Rotating +X by
    # atan2(forward_run, drop) tips it up to aim at the target point.
    pitch = math.degrees(math.atan2(aim_y - y_cam, TOP_CAM_HEIGHT))
    return {"pos": [0.0, y_cam, TOP_CAM_HEIGHT],
            "euler_deg": [pitch, 0.0, 0.0], "fitted": False}


# i2rt's yam.xml gives every link a different debug colour (yellow, pink,
# olive...). Useful for reading a kinematic chain, wrong for anything that has
# to resemble the real arms. "dark" matches the real hardware's near-black
# anodising; "translucent" follows the SPD paper, which makes the arms
# see-through so they occlude less of the scene during collection.
# Arm colour. i2rt's four YAM variants (yam, yam_pro, big_yam, yam_ultra) all
# ship the same six arbitrary debug colours -- purple base, tan, blue, olive,
# grey, pink -- with no materials or textures, and each link is a single STL, so
# a link and its own joint housing cannot be coloured separately. Flat grey it
# is; "raw" keeps i2rt's colours and "translucent" follows the SPD paper.
ARM_STYLES = {
    "raw": None,
    "grey": (0.55, 0.55, 0.56, 1.0),
    "translucent": (0.60, 0.62, 0.66, 0.35),
}


def build(table_color: str = "asshot", light_scale: float = 1.0,
          arm_style: str = "grey") -> ET.ElementTree:
    intr_path = _HERE / "camera_intrinsics.json"
    if not intr_path.exists():
        raise SystemExit(f"{intr_path} missing -- run tools/sim_dump_intrinsics.py first")
    intrinsics = json.loads(intr_path.read_text())

    pose_path = _HERE / "top_camera_pose.json"
    top_pose = (json.loads(pose_path.read_text()) if pose_path.exists()
                else _default_top_pose())

    arm_src = _arm_source()
    mj = ET.Element("mujoco", model="spd_bi_yam")
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true", eulerseq="xyz")
    # Physics settings follow the SPD paper: 480 Hz, implicitfast integrator,
    # elliptic friction cones, one no-slip iteration.
    # Physics per the SPD paper: 480 Hz, implicitfast, elliptic friction cones,
    # one no-slip iteration. impratio is ours: MuJoCo's default of 1 leaves
    # frictional contacts soft enough that a stacked cube creeps sideways
    # (~13 mm in 3 s), and raising it stiffens friction relative to normal
    # force, which is what stacking needs.
    ET.SubElement(mj, "option", timestep=f"{1/480:.10g}", integrator="implicitfast",
                  cone="elliptic", noslip_iterations="1", impratio="10")
    visual = ET.SubElement(mj, "visual")
    # Offscreen buffer must be at least the camera resolution we render at.
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")

    # ── assets ──────────────────────────────────────────────────────────
    asset = ET.SubElement(mj, "asset")
    for mesh in arm_src.findall("./asset/mesh"):
        asset.append(deepcopy(mesh))
    ET.SubElement(asset, "material", name="table_asshot",
                  rgba=" ".join(f"{v:.6g}" for v in TABLE_RGBA_ASSHOT),
                  specular="0.35", shininess="0.55", reflectance="0.05")
    ET.SubElement(asset, "material", name="table_neutral",
                  rgba=" ".join(f"{v:.6g}" for v in TABLE_RGBA_NEUTRAL),
                  specular="0.35", shininess="0.55", reflectance="0.05")
    ET.SubElement(asset, "material", name="floor_mat", rgba="0.62 0.5 0.36 1",
                  specular="0.1", shininess="0.1")
    ET.SubElement(asset, "material", name="frame_mat", rgba="0.25 0.25 0.27 1")

    table_mat = "table_asshot" if table_color == "asshot" else "table_neutral"

    world = ET.SubElement(mj, "worldbody")
    # Light levels are calibrated so a flat patch of tabletop renders at the
    # brightness actually measured off greenscreen/top.png -- see
    # tools/sim_calibrate_lighting.py. Without this the render sits ~6% hot and
    # every colour comparison inherits the offset.
    c = 0.55 * light_scale
    f = 0.30 * light_scale
    ET.SubElement(world, "light", name="ceiling", pos="0 0 2.4", dir="0 0 -1",
                  directional="false", diffuse=f"{c:.4g} {c:.4g} {c:.4g}",
                  specular="0.15 0.15 0.15")
    ET.SubElement(world, "light", name="fill", pos="0.9 -1.2 2.0", dir="-0.4 0.5 -1",
                  directional="false", diffuse=f"{f:.4g} {f:.4g} {f:.4g}")

    # Floor, at the real distance below the tabletop so the strip of floor
    # visible beyond the far edge lands in the right place in the render.
    ET.SubElement(world, "geom", name="floor", type="plane",
                  pos=f"0 0 {-TABLE_HEIGHT:.6g}", size="4 4 0.1",
                  material="floor_mat")

    # ── table ───────────────────────────────────────────────────────────
    half = TABLE_SIZE / 2
    tbody = ET.SubElement(world, "body", name="table", pos="0 0 0")
    ET.SubElement(tbody, "geom", name="table_top", type="box",
                  pos=f"0 0 {-TABLE_THICKNESS/2:.6g}",
                  size=f"{half:.6g} {half:.6g} {TABLE_THICKNESS/2:.6g}",
                  material=table_mat)
    leg = 0.04
    leg_h = (TABLE_HEIGHT - TABLE_THICKNESS) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            ET.SubElement(tbody, "geom",
                          name=f"table_leg_{'n' if sy < 0 else 'f'}{'l' if sx < 0 else 'r'}",
                          type="box",
                          pos=f"{sx*(half-0.06):.6g} {sy*(half-0.06):.6g} "
                              f"{-TABLE_THICKNESS - leg_h:.6g}",
                          size=f"{leg:.6g} {leg:.6g} {leg_h:.6g}",
                          material="frame_mat")

    # ── task objects ────────────────────────────────────────────────────
    # Free bodies, so they can actually be pushed and picked up. PLA on
    # melamine is fairly slippery; 0.4 sliding friction is a reasonable stand-in.
    mass = _cube_mass()
    for name, cx, cy, cz in cube_poses():
        b = ET.SubElement(world, "body", name=name,
                          pos=f"{cx:.6g} {cy:.6g} {cz:.6g}")
        ET.SubElement(b, "freejoint", name=f"{name}_free")
        ET.SubElement(b, "geom", name=f"{name}_geom", type="box",
                      size=" ".join([f"{CUBE_EDGE/2:.6g}"] * 3),
                      rgba=" ".join(f"{v:.4g}" for v in CUBE_COLOURS[name]),
                      mass=f"{mass:.6g}", friction="0.4 0.005 0.0001",
                      solimp="0.95 0.99 0.001", solref="0.005 1")

    # ── arms ────────────────────────────────────────────────────────────
    # Bases sit at the near edge, ARM_BASE_SEPARATION apart, each yawed to
    # face across the table (+Y). Left arm is at -X, right at +X.
    for side, sx in (("left", -1), ("right", +1)):
        prefix = f"{side}_"
        mount = ET.SubElement(
            world, "body", name=f"{side}_mount",
            pos=f"{sx * ARM_BASE_SEPARATION / 2:.6g} {ARM_BASE_Y:.6g} 0",
            euler=f"0 0 {math.radians(ARM_MOUNT_YAW_DEG):.10g}")
        for child in arm_src.find("worldbody"):
            node = deepcopy(child)
            _prefix_tree(node, prefix)
            # Gravity compensation on every arm body. i2rt's yam.xml caps each
            # joint's actuator at +-10 Nm, which is nowhere near enough to hold
            # the arm up: a position-controlled arm sags ~0.6 rad at joint 2 and
            # cannot lift at all. The real robot does not fight gravity either --
            # get_yam_robot runs zero_gravity_mode=True and feeds gravity torques
            # forward -- so cancelling it here matches the hardware rather than
            # papering over the force limit.
            for b in ([node] if node.tag == "body" else []) + node.findall(".//body"):
                b.set("gravcomp", "1")
            mount.append(node)
        # Name EVERY geom in the arm subtree. i2rt's yam.xml leaves its link
        # geoms unnamed, and an unnamed geom cannot be addressed afterwards:
        # segmentation masks, arm hiding and per-arm styling all key off the
        # name prefix, and each of those silently matches nothing (an empty
        # arm mask, not an error) if the geoms stay anonymous.
        seen: dict[str, int] = {}
        for i, g in enumerate(mount.iter("geom")):
            if g.get("name"):
                continue
            stem = g.get("mesh") or f"geom{i}"
            n = seen.get(stem, 0)
            seen[stem] = n + 1
            g.set("name", f"{prefix}{stem}" if n == 0 else f"{prefix}{stem}_{n}")
        # Hide the arm's debug sites. i2rt's gripper XML ships a red tcp_site and
        # a green grasp_site, and MuJoCo draws sites by default -- they show up
        # as coloured dots on every rendered frame, i.e. as features in the
        # training images that have no counterpart on the real robot. Alpha only
        # affects drawing, so the sites stay fully usable for IK and for the
        # grasp checks that reference grasp_site by name.
        for st in mount.iter("site"):
            rgba = (st.get("rgba") or "0 0 0 1").split()
            st.set("rgba", " ".join(rgba[:3] + ["0"]) if len(rgba) >= 3
                   else "0 0 0 0")

        # Coulomb friction on the arm joints, per yam.yml.
        for jt in mount.iter("joint"):
            nm = (jt.get("name") or "")
            for j in range(1, 7):
                if nm == f"{prefix}joint{j}":
                    jt.set("frictionloss", f"{ARM_FRICTIONLOSS[j-1]:.6g}")
        rgba = ARM_STYLES.get(arm_style)
        if rgba is not None:
            for g in mount.iter("geom"):
                g.set("rgba", " ".join(f"{v:.4g}" for v in rgba))
        # Gripper pads: explicit collision slabs, with the fingertip meshes
        # demoted to visual-only.
        #
        # MuJoCo collides a mesh geom as its CONVEX HULL, and these fingertips
        # are concave prongs -- the hull is 3.1x the mesh volume, filling in the
        # V between them. The result is a grasp that works but looks wrong: with
        # a 35 mm cube held, one finger carried 2.7 N of contact force while its
        # nearest *visible* surface was 6.8 mm away from the cube. The cube was
        # being held by geometry that is not drawn.
        #
        # The real contact surface is a flat 27 x 17 mm pad on each finger,
        # centred at the grasp-site height (measured: gripper-frame y = +-47.05 mm,
        # z spanning -144.5..-127.2 mm). These boxes sit exactly there, 3 mm
        # thick, so contact happens within ~1.5 mm of the surface you can see.
        #
        # Pad friction is rubber-like rather than MuJoCo's default 1.0: the pads
        # are small and the contact is nearly point-like, and at 1.0 the grasp
        # was a knife edge -- only one force cap in six held a cube through a
        # lift, non-monotonically. i2rt ships a FLEXIBLE_4310 "soft tips"
        # variant, so a rubber coefficient is faithful as well as forgiving.
        PAD = {
            "tip_left":  ((-0.0794, +0.0242, -0.0446), (0.0087, 0.0135, 0.0015)),
            "tip_right": ((-0.0239, -0.0792, -0.0446), (0.0135, 0.0087, 0.0015)),
        }
        for g in mount.iter("geom"):
            mesh = g.get("mesh") or ""
            if mesh in ("tip_left", "tip_right"):
                # visual only -- its convex hull must not collide
                g.set("contype", "0")
                g.set("conaffinity", "0")
        for body in mount.iter("body"):
            bn = body.get("name") or ""
            for mesh, (pos, size) in PAD.items():
                if bn == f"{prefix}{mesh}":
                    ET.SubElement(body, "geom",
                                  name=f"{prefix}{mesh}_pad", type="box",
                                  pos=" ".join(f"{v:.6g}" for v in pos),
                                  size=" ".join(f"{v:.6g}" for v in size),
                                  friction="1.6 0.02 0.002",
                                  # Invisible: these exist only to collide. rgba
                                  # affects rendering alone, so alpha 0 keeps the
                                  # contact behaviour and keeps a bright debug
                                  # tab out of every training image -- the
                                  # fingertip mesh is what should be seen.
                                  rgba="0.6 0.62 0.66 0",
                                  solimp="0.95 0.99 0.001", solref="0.005 1")

    # equality / contact blocks, namespaced per side
    eq_src, ct_src = arm_src.find("equality"), arm_src.find("contact")
    if eq_src is not None:
        eq = ET.SubElement(mj, "equality")
        for side in ("left", "right"):
            for node in eq_src:
                n = deepcopy(node)
                _prefix_refs(n, f"{side}_")
                for attr in ("joint1", "joint2"):
                    if n.get(attr) and not n.get(attr).startswith(f"{side}_"):
                        n.set(attr, f"{side}_" + n.get(attr))
                eq.append(n)
    ct = ET.SubElement(mj, "contact")
    if ct_src is not None:
        for side in ("left", "right"):
            for node in ct_src:
                n = deepcopy(node)
                for attr in ("body1", "body2"):
                    if n.get(attr):
                        n.set(attr, f"{side}_" + n.get(attr))
                ct.append(n)
    # Exclude self-collision between adjacent arm links.
    #
    # The mount<->link1 pair is the one that actually bites, and it is subtle:
    # MuJoCo filters parent/child geom pairs automatically, but skips that
    # filter when the parent's *weld* is the world body. The mount is a static
    # body, so it welds to the world, so base and link1 are collision-checked --
    # and their meshes overlap by ~6 mm at every joint angle. The result is eight
    # permanent contacts fighting joint 1: it saturates its +-10 Nm limit at ~24 Nm
    # of demand and simply refuses to rotate, so the arm lags the IK target by
    # ~0.3 rad and teleop feels like the IK is broken when it is not.
    #
    # i2rt's yam.xml has the same latent overlap (its base geom sits directly in
    # worldbody, which is the same "parent is world" case); it never shows up
    # there because i2rt's SimRobot does not simulate contact.
    #
    # The remaining pairs are already filtered by MuJoCo, and are listed for
    # robustness against future re-parenting.
    for side in ("left", "right"):
        chain = ["mount", "link1", "link2", "link3", "link4", "link5", "link6",
                 "gripper"]
        for a, b in zip(chain, chain[1:]):
            ET.SubElement(ct, "exclude", body1=f"{side}_{a}", body2=f"{side}_{b}")

    # ── cameras ─────────────────────────────────────────────────────────
    # Top camera: a free-standing body between the arms. MuJoCo cameras look
    # down their own -Z with +Y up, so a camera with zero euler stares at the
    # floor; the fitted euler tilts it up to frame the table.
    top_intr = intrinsics.get("top")
    if top_intr is None:
        raise SystemExit("no 'top' entry in camera_intrinsics.json")
    px, py, pz = top_pose["pos"]
    # The compiler is in radians (the i2rt joint ranges demand it), and that
    # applies to euler too -- degrees here would be silently wrapped.
    ex, ey, ez = (math.radians(v) for v in top_pose["euler_deg"])
    cam_body = ET.SubElement(world, "body", name="top_cam_mount",
                             pos=f"{px:.6g} {py:.6g} {pz:.6g}",
                             euler=f"{ex:.10g} {ey:.10g} {ez:.10g}")
    # No housing geom on purpose: a box at the camera casts a small hard shadow
    # onto the tabletop, right inside the camera's own view. viser already draws
    # the camera as a frustum, so the box bought nothing and cost an artifact.
    ET.SubElement(cam_body, "camera", name="top", pos="0 0 0",
                  quat="1 0 0 0", **_camera_intrinsic_attrs(top_intr))

    # Wrist cameras.
    #
    # DELIBERATELY APPROXIMATE, and worth understanding why that is defensible.
    # The intrinsics are exact (read off each D405's EEPROM). The mount is not
    # calibrated: attempts to recover it from imagery topped out at 0.34
    # finger-silhouette IoU, because 6 DoF from a clipped silhouette of two
    # near-symmetric jaws is badly conditioned. squint's YAM.md documents the
    # real station's extrinsics, but in a URDF whose wrist is decomposed
    # differently (its link_6 carries the fingers at +-37 mm; ours at +-24 mm),
    # and transferring them through the frame-independent distances scored worse
    # still (0.29).
    #
    # So this is chosen for *view usefulness* rather than fidelity: workspace
    # filling ~47% of the frame, the jaws occupying ~6% and entirely in the
    # lower part of the image -- which is what a real wrist view looks like.
    # Anchored to squint's ~40 deg bracket tilt, and to a standoff calibrated
    # against the real wrist frames by image composition rather than by pose:
    # in calib/ the jaws fill 25.9% of the frame and the table 63.7%, with 49%
    # of the jaw pixels low in the image. At a 50 mm standoff the sim gives
    # 27.0% / 70.5% / 52% -- close on all three. An 80 mm standoff put the jaws
    # at only 5.7%, i.e. the lens much further from the fingers than reality.
    #
    # Known limitation: the block being grasped stays just below the frame edge
    # at every standoff, so the wrist views do not show the object itself. That
    # is tolerable here because the top camera -- which IS calibrated, to
    # 0.409 px -- carries the object's identity and position, and the wrist
    # views contribute close-range context. Worth revisiting if fine grasp
    # alignment turns out to be the failure mode.
    #
    # This is a reasonable thing to ship because the SPD recipe is not zero-shot:
    # it fine-tunes on 1-2 hours of real data per task and swaps table and
    # background textures as augmentation, so it never depended on a pixel-exact
    # camera match. An approximately-right pose is the kind of error fine-tuning
    # absorbs. A camera *inside the housing* -- which is what the previous guess
    # produced -- is not, because it carries no information about the scene.
    #
    # Both arms share one transform, matching squint's note that the two wrist
    # D405s have byte-identical extrinsics on the real station.
    WRIST_CAM_POS = (0.0000, -0.0438, -0.0305)
    WRIST_CAM_QUAT = (-0.3596, -0.1309, 0.3160, 0.8682)
    for side in ("left", "right"):
        cam_id = f"{side}_wrist"
        wi = intrinsics.get(cam_id)
        if wi is None:
            continue
        gripper = None
        for b in mj.iter("body"):
            if b.get("name") == f"{side}_gripper":
                gripper = b
                break
        if gripper is None:
            continue
        ET.SubElement(gripper, "camera", name=cam_id,
                      pos=" ".join(f"{v:.6g}" for v in WRIST_CAM_POS),
                      quat=" ".join(f"{v:.6g}" for v in WRIST_CAM_QUAT),
                      **_camera_intrinsic_attrs(wi))

    # ── actuators ───────────────────────────────────────────────────────
    act = ET.SubElement(mj, "actuator")
    for side in ("left", "right"):
        for j in range(1, 7):
            ET.SubElement(act, "position", name=f"{side}_joint{j}",
                          joint=f"{side}_joint{j}",
                          kp=f"{ARM_KP[j-1]:.6g}", kv=f"{ARM_KD[j-1]:.6g}")
        # joint8 follows joint7 through the equality constraint.
        #
        # Stiffness matters here and a soft guess does not survive contact. A
        # position actuator delivers kp * error, so kp=80 N/m against the ~0.027 m
        # error a held cube leaves gave only ~2 N at the joint and ~0.9 N at the
        # fingers: enough to hold a 17 g cube at rest, but it broke loose the
        # moment the arm accelerated, and the jaws then closed on nothing.
        #
        # The real gripper is far stiffer. linear_4310.yml has motor_kp = 20
        # Nm/rad driving motor_stroke 6.57 rad over gripper_stroke 0.096 m, i.e.
        # a 68.4 rad/m reduction, which reflects to roughly 9e4 N/m at the jaw.
        # Using that verbatim would slam 2 kN into a PLA cube, so the stiffness
        # is kept firm but the force is capped at a plausible grasp load --
        # the hardware has its own force limiter (clog_force_threshold) doing
        # the same job. 6 N sits mid-band: with the rubber pad friction above,
        # a sweep held the cube through a lift anywhere from 2.5 N to 10 N, so
        # this is comfortably inside the robust range rather than tuned to an
        # edge. Too much force is its own failure -- 25 N extrudes a 35 mm cube
        # out from between the angled fingers.
        ET.SubElement(act, "position", name=f"{side}_gripper",
                      joint=f"{side}_joint7", kp="3000", kv="30",
                      forcerange="-6 6", ctrlrange="0 0.0475")

    _add_keyframes(mj)

    ET.indent(mj, space="  ")
    return ET.ElementTree(mj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-color", choices=("asshot", "neutral"), default="asshot",
                    help="asshot = measured cyan from greenscreen/top.png "
                         "(pixel match); neutral = grey-world corrected albedo")
    ap.add_argument("--light-scale", type=float, default=LIGHT_SCALE,
                    help="scales both lights; calibrated by "
                         "tools/sim_calibrate_lighting.py")
    ap.add_argument("--arm-style", choices=tuple(ARM_STYLES), default="grey",
                    help="grey = flat neutral; raw = i2rt per-link debug "
                         "colours; translucent follows the SPD paper")
    ap.add_argument("--out", default=str(_HERE / "spd_scene.xml"))
    args = ap.parse_args()

    tree = build(args.table_color, args.light_scale, args.arm_style)
    dest = Path(args.out)
    tree.write(dest, encoding="unicode", xml_declaration=False)
    dest.write_text(dest.read_text() + "\n")
    print(f"wrote {dest}  (table_color={args.table_color}, "
          f"arm_style={args.arm_style}, light_scale={args.light_scale})")

    import mujoco
    m = mujoco.MjModel.from_xml_path(str(dest))
    print(f"loads OK: nq={m.nq} nu={m.nu} nbody={m.nbody} ncam={m.ncam} ngeom={m.ngeom}")
    for i in range(m.ncam):
        print(f"  camera {i}: {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)}"
              f"  res={m.cam_resolution[i]}  fovy={m.cam_fovy[i]:.4f}"
              f"  sensorsize={m.cam_sensorsize[i]}  intrinsic={m.cam_intrinsic[i]}")


if __name__ == "__main__":
    main()
