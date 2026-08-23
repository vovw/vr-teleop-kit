"""Prove the MuJoCo cameras reproduce the measured pixel intrinsics exactly.

Writing fx/fy/principal into MJCF is only half the job -- the sign and scale
conventions have to be right too, and a silent mistake there shows up as a
scene that looks plausible but is quietly mis-scaled. So instead of trusting
the XML, this renders a probe: put a small sphere at a known world point,
render, take the centroid of its pixels, and compare against the pinhole
projection computed from the measured intrinsic matrix and MuJoCo's own
reported camera pose.

Agreement to a fraction of a pixel means a render of this scene and a frame
from the real camera are directly comparable, which is the whole premise of
the pixel match.

    DISPLAY=:0 MUJOCO_GL=glfw python tools/sim_verify_intrinsics.py
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

PROBE_RGBA = "1 0 0 1"
PROBE_RADIUS = 0.010


def project(K: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray,
            pts: np.ndarray) -> np.ndarray:
    """Pinhole-project world points using MuJoCo's camera convention.

    A MuJoCo camera looks down its own -Z with +Y up, so depth in front of the
    lens is -Z_cam, and image rows increase as Y_cam decreases.
    """
    rel = pts - cam_pos                     # world offset from the lens
    cam = rel @ cam_mat                     # into camera axes (cam_mat columns are the axes)
    depth = -cam[:, 2]
    u = K[0, 2] + K[0, 0] * cam[:, 0] / depth
    v = K[1, 2] - K[1, 1] * cam[:, 1] / depth
    return np.stack([u, v], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(_REPO_ROOT / "sim" / "spd_scene.xml"))
    ap.add_argument("--camera", default="top")
    args = ap.parse_args()

    import mujoco

    intr = json.loads((_REPO_ROOT / "sim" / "camera_intrinsics.json").read_text())[args.camera]
    W, H = intr["width"], intr["height"]
    K = np.array([[intr["fx"], 0, intr["ppx"]],
                  [0, intr["fy"], intr["ppy"]],
                  [0, 0, 1]])

    # Inject a probe sphere, and hide everything that could occlude it.
    # The probe material is fully emissive on purpose: a normally-shaded
    # sphere is bright on top and dark underneath, so a colour threshold
    # clips the dark side and drags the centroid upward by ~1 px -- enough
    # to masquerade as a vertical intrinsics error.
    root = ET.parse(args.scene).getroot()
    asset = root.find("asset")
    ET.SubElement(asset, "material", name="__probe_mat", rgba=PROBE_RGBA,
                  emission="1", specular="0", shininess="0")
    world = root.find("worldbody")
    ET.SubElement(world, "body", name="__probe", pos="0 0 0").append(
        ET.Element("geom", {"name": "__probe_g", "type": "sphere",
                            "size": str(PROBE_RADIUS), "material": "__probe_mat",
                            "contype": "0", "conaffinity": "0"}))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)

    # Hide the arms so the probe is never occluded by a link.
    probe_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "__probe_g")
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith(("left_", "right_")):
            model.geom_rgba[gid, 3] = 0.0
    probe_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "__probe")

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    renderer = mujoco.Renderer(model, H, W)

    # Probe points spread over the tabletop, plus a couple above it.
    grid = [(x, y, z)
            for z in (0.02, 0.25)
            for y in (-0.5, -0.1, 0.3, 0.65)
            for x in (-0.55, 0.0, 0.55)]

    print(f"camera '{args.camera}'  {W}x{H}  "
          f"fx={intr['fx']:.4f} fy={intr['fy']:.4f} "
          f"ppx={intr['ppx']:.4f} ppy={intr['ppy']:.4f}")
    print(f"{'world (x,y,z)':>22} | {'predicted (u,v)':>17} | "
          f"{'rendered (u,v)':>17} | err px")
    print("-" * 76)

    errs = []
    for pt in grid:
        model.body_pos[probe_bid] = pt
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam_id)
        img = renderer.render()

        cam_pos = data.cam_xpos[cam_id].copy()
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
        pred = project(K, cam_pos, cam_mat, np.array([pt], dtype=float))[0]

        # The probe is the only saturated-red thing in the frame.
        r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
        m = (r > 110) & (r - g > 55) & (r - b > 55)
        if m.sum() < 8:
            print(f"{str(tuple(pt)):>22} | {pred[0]:8.2f},{pred[1]:8.2f} | "
                  f"{'not visible':>17} |   --")
            continue
        ys, xs = np.nonzero(m)
        # A probe whose silhouette runs off the edge has a truncated centroid;
        # that measures clipping, not intrinsics, so it is reported and skipped.
        if (xs.min() == 0 or ys.min() == 0
                or xs.max() == img.shape[1] - 1 or ys.max() == img.shape[0] - 1):
            print(f"{str(tuple(pt)):>22} | {pred[0]:8.2f},{pred[1]:8.2f} | "
                  f"{'clipped at edge':>17} |   --")
            continue
        meas = np.array([xs.mean() + 0.5, ys.mean() + 0.5])  # pixel centres
        err = float(np.hypot(*(meas - pred)))
        errs.append(err)
        print(f"{str(tuple(pt)):>22} | {pred[0]:8.2f},{pred[1]:8.2f} | "
              f"{meas[0]:8.2f},{meas[1]:8.2f} | {err:6.3f}")

    if errs:
        e = np.array(errs)
        print("-" * 76)
        print(f"{len(e)} visible probes: mean {e.mean():.3f} px, "
              f"median {np.median(e):.3f} px, max {e.max():.3f} px")
        verdict = "EXACT" if e.max() < 0.5 else "MISMATCH"
        print(f"verdict: {verdict} (sub-pixel agreement expected; "
              f"centroid of a sphere is itself ~0.3px noisy)")


if __name__ == "__main__":
    main()
