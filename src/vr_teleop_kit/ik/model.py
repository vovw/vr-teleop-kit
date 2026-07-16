"""Mujoco model construction for the I2RT YAM arm.

Combines the YAM arm MJCF (`yam.xml`) with the linear_4310 gripper MJCF
the same way i2rt's `combine_arm_and_gripper_xml` does — the arm's last
body (`link6`) gets the gripper mount transform, and the gripper subtree
(body `gripper` + finger sliders `joint7`/`joint8`) is appended under it.
The combined model has nq = 8: arm joints 1-6 at qpos[0:6], the two
finger sliders at qpos[6:8].

The spec is then decorated with two named sites the IK pipeline needs:

  tool0      — the end-effector target, placed at the gripper's grasp
               point (between the fingertips; same pose as the gripper
               XML's `grasp_site`).
  j4_anchor  — the position-task anchor used by the decoupled IK. Placed
               on link3 (upstream of joint 4 → wrist-invariant) and
               offset 10 cm past joint 4's pivot along the link3→link4
               direction to keep the anchor off the joint-1 axis at
               folded poses.

The model files are deliberately NOT vendored. Clone the i2rt repository
(https://github.com/i2rt-robotics/i2rt) and point at
`i2rt/robot_models/arm/yam/yam.xml`, either explicitly or via the
`YAM_XML` environment variable (see `resolve_model_path`); a clone at
`./i2rt` inside this repo is picked up automatically. The gripper MJCF
is located relative to the arm MJCF inside that clone.

The IK math itself lives in decoupled_ik.py.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

# Environment variable consulted when no explicit model path is given.
YAM_XML_ENV = "YAM_XML"

# Default location of the arm MJCF inside an i2rt clone, tried relative
# to the current directory and to this repo's root when neither an
# explicit path nor the environment variable is set.
_I2RT_ARM_XML_REL = Path("i2rt/i2rt/robot_models/arm/yam/yam.xml")

# Gripper MJCF, relative to the `robot_models` directory that also holds
# the arm MJCF (i.e. `<clone>/i2rt/robot_models/`).
_GRIPPER_XML_REL = Path("gripper/linear_4310/linear_4310.xml")

# Mount transform for the linear_4310 gripper on the YAM arm — pos/quat of
# the arm's last body (link6) and the axis of joint6. Transcribed from
# `i2rt/robots/config/linear_4310.yml` (last_joint_mount.yam); i2rt applies
# the same values when it assembles the hardware model.
_MOUNT_POS = "2.39858e-07 -0.0419481 0.0404996"
_MOUNT_QUAT = "0.499998 -0.5 -0.5 -0.500002"
_MOUNT_AXIS = "0 0 -1"

# Grasp point in the gripper body frame — between the fingertips, same
# pose as the gripper XML's `grasp_site`. Used for the `tool0` site.
_TOOL0_POS = [0.0, 0.0, -0.1347]
_TOOL0_QUAT = [0.0, 1.0, 0.0, 0.0]

DEFAULT_Q_REST = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])


def resolve_model_path(explicit: str | Path | None = None) -> Path:
    """Resolve the YAM arm MJCF path: an explicit argument wins, then the
    `YAM_XML` environment variable, then an `i2rt` clone at the current
    directory or this repo's root. Raises with setup instructions when
    nothing is configured or the file is missing."""
    raw = explicit or os.environ.get(YAM_XML_ENV)
    if raw:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"YAM arm MJCF not found at {path}")
        return path
    for root in (Path.cwd(), Path(__file__).resolve().parents[3]):
        candidate = root / _I2RT_ARM_XML_REL
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No YAM model configured. Pass model_path=... or set the YAM_XML "
        "environment variable to .../i2rt/robot_models/arm/yam/yam.xml "
        "(clone https://github.com/i2rt-robotics/i2rt)."
    )


def _find_deepest_body(element: ET.Element) -> ET.Element:
    """Leaf body of the arm's kinematic chain (first-child descent) —
    same rule i2rt uses to locate where the gripper mounts."""
    current = element
    while True:
        child_bodies = [c for c in current if c.tag == "body"]
        if not child_bodies:
            return current
        current = child_bodies[0]


def _absolutize_mesh_paths(root: ET.Element, xml_path: Path) -> ET.Element | None:
    """Rewrite the <asset> mesh file paths to absolute (resolving the
    compiler meshdir) so the combined spec compiles regardless of cwd.
    Returns the <asset> element (or None)."""
    base_dir = xml_path.resolve().parent
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    asset = root.find("asset")
    if asset is not None:
        for child in asset:
            f = child.get("file")
            if f and not os.path.isabs(f):
                child.set("file", str((base_dir / meshdir / f).resolve()))
    if compiler is not None and compiler.get("meshdir"):
        del compiler.attrib["meshdir"]
    return asset


def _combined_arm_gripper_xml(arm_xml: Path, gripper_xml: Path) -> str:
    """Merge the gripper MJCF into the arm MJCF (mount transform on the
    arm's last body, gripper subtree appended under it, assets/equality/
    contact sections merged) and return the combined XML string."""
    arm_root = ET.parse(arm_xml).getroot()
    worldbody = arm_root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"no <worldbody> in {arm_xml}")
    last_link = _find_deepest_body(worldbody)
    last_link.set("pos", _MOUNT_POS)
    last_link.set("quat", _MOUNT_QUAT)
    last_joint = last_link.find("joint")
    if last_joint is not None:
        last_joint.set("axis", _MOUNT_AXIS)

    arm_asset = _absolutize_mesh_paths(arm_root, arm_xml)

    grip_root = ET.parse(gripper_xml).getroot()
    grip_asset = _absolutize_mesh_paths(grip_root, gripper_xml)
    grip_body = grip_root.find(".//body[@name='gripper']")
    if grip_body is None:
        raise RuntimeError(f"no <body name='gripper'> in {gripper_xml}")

    if grip_asset is not None:
        if arm_asset is None:
            arm_asset = ET.SubElement(arm_root, "asset")
        existing = {(c.tag, c.get("name")) for c in arm_asset}
        for child in grip_asset:
            if (child.tag, child.get("name")) not in existing:
                arm_asset.append(deepcopy(child))

    last_link.append(deepcopy(grip_body))

    for tag in ("equality", "contact"):
        grip_section = grip_root.find(tag)
        if grip_section is None:
            continue
        arm_section = arm_root.find(tag)
        if arm_section is None:
            arm_section = ET.SubElement(arm_root, tag)
        for child in grip_section:
            arm_section.append(deepcopy(child))

    return ET.tostring(arm_root, encoding="unicode")


def build_model_with_tool0_site(
    model_path: str | Path | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    arm_xml = resolve_model_path(model_path)
    gripper_xml = arm_xml.resolve().parents[2] / _GRIPPER_XML_REL
    if not gripper_xml.exists():
        raise FileNotFoundError(
            f"gripper MJCF not found at {gripper_xml} — expected the arm "
            "MJCF to live inside an i2rt clone at "
            ".../i2rt/robot_models/arm/yam/yam.xml"
        )
    spec = mujoco.MjSpec.from_string(_combined_arm_gripper_xml(arm_xml, gripper_xml))

    gripper = spec.body("gripper")
    if gripper is None:
        raise RuntimeError("gripper body not found in combined spec")
    gripper.add_site(name="tool0", pos=_TOOL0_POS, quat=_TOOL0_QUAT)

    # Position-task anchor. On link3 (upstream of joint 4 → fully
    # wrist-invariant), 10 cm past joint 4's pivot along the link3→link4
    # direction. Computed from link4's mount position in the spec rather
    # than transcribed, so it tracks the model files.
    link3 = spec.body("link3")
    link4 = spec.body("link4")
    if link3 is None or link4 is None:
        raise RuntimeError("link3/link4 body not found in combined spec")
    j4_pivot = np.asarray(link4.pos, dtype=float)
    anchor = j4_pivot + 0.10 * j4_pivot / np.linalg.norm(j4_pivot)
    link3.add_site(
        name="j4_anchor",
        pos=anchor.tolist(),
        size=[0.015, 0.0, 0.0],     # 1.5 cm sphere
        rgba=[1.0, 0.5, 0.0, 1.0],  # orange
    )
    model = spec.compile()
    return model, mujoco.MjData(model)
