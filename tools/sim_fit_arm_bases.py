"""Fit each arm's base pose against the calibrated top camera.

The top camera is pinned to 0.4 px, and the recording pairs every frame with
joint angles, which makes the arm bases measurable rather than assumed: put the
sim arms at the recorded joint angles, render, and move the bases until the
rendered arms land on the real ones.

This matters more than it sounds. The base pose was originally a guess -- 0.50 m
apart at the near edge, yawed -90 deg -- and the guess was ~180 deg wrong. At the
recorded joint angles it placed the grippers at y ~ -1.1, reaching away from the
table and behind the camera, so *zero* arm pixels rendered where reality has
nearly ten thousand. The -90 deg had been "checked" at j2=0, which is a joint
limit and an unrepresentative configuration; the real poses sit at j2 ~ 1.0-2.2
and reach the other way entirely.

Arms segment cleanly in the top view despite being white on a white table: the
links read markedly brighter and bluer than the melamine (measured B-R of +49
against -7), which separates them where brightness alone does not.

Base bodies are moved through model.body_pos / body_quat rather than by
regenerating the scene, so an optimisation step costs a render instead of an
XML compile.

    DISPLAY=:0 MUJOCO_GL=glfw python tools/sim_fit_arm_bases.py

Writes sim/arm_base_poses.json; re-run sim/build_scene.py afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

_REPO_ROOT = Path(__file__).resolve().parents[1]


def observed_arm_masks(img: np.ndarray) -> dict[str, np.ndarray]:
    """Split the visible arm pixels into left / right by component centroid.

    A YAM arm is two-tone: white links plus black joint housings and a black
    distal assembly. Keying on the white alone (as a first attempt did) finds
    one link and misses most of the arm, so both extremes are taken -- bright
    and blue-ish for the links (measured B-R of +49 against the tabletop's -7),
    very dark for the housings.

    Restricted to the lower part of the frame: the wooden floor beyond the far
    edge is also dark, and would otherwise be swallowed by the dark term.
    """
    h, w = img.shape[:2]
    B, G, R = (img[:, :, i].astype(int) for i in range(3))
    lum = img.mean(axis=2)
    bright_links = (lum > 150) & ((B - R) > 25)
    dark_parts = lum < 90
    m = ((bright_links | dark_parts)).astype(np.uint8) * 255
    m[: int(0.35 * h), :] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
    out = {"left": np.zeros_like(m), "right": np.zeros_like(m)}
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 400:
            continue
        side = "left" if cent[i][0] < w / 2 else "right"
        out[side][lab == i] = 255
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=str(_REPO_ROOT / "calib"))
    ap.add_argument("--scene", default=str(_REPO_ROOT / "sim" / "spd_scene.xml"))
    ap.add_argument("--out", default=str(_REPO_ROOT / "sim" / "arm_base_poses.json"))
    args = ap.parse_args()

    import mujoco
    import sys
    sys.path.insert(0, str(_REPO_ROOT / "tools"))
    from sim_compare_top import distortion_maps

    intr = json.loads((_REPO_ROOT / "sim" / "camera_intrinsics.json").read_text())["top"]
    W, H = intr["width"], intr["height"]
    maps = distortion_maps(intr, cache=_REPO_ROOT / "sim" / "top_distortion_map.npz")

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    renderer = mujoco.Renderer(model, H, W)
    renderer.enable_segmentation_rendering()

    geom_ids = {"left": [], "right": []}
    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        for side in ("left", "right"):
            if nm.startswith(f"{side}_"):
                geom_ids[side].append(gid)
    mount = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_mount")
             for s in ("left", "right")}

    poses = []
    for pd in sorted(Path(args.calib).glob("pose_*")):
        top = pd / "top.png"
        if not top.exists():
            continue
        img = cv2.imread(str(top))
        poses.append({"name": pd.name,
                      "joints": json.loads((pd / "joints.json").read_text()),
                      "obs": observed_arm_masks(img)})
    if not poses:
        raise SystemExit("no poses with top.png found")
    for p in poses:
        print(f"  {p['name']}: observed arm px "
              f"left={int((p['obs']['left']>127).sum())} "
              f"right={int((p['obs']['right']>127).sum())}")

    def set_base(side, x, y, z, yaw):
        model.body_pos[mount[side]] = (x, y, z)
        h = yaw / 2.0
        model.body_quat[mount[side]] = (math.cos(h), 0.0, 0.0, math.sin(h))

    def render_masks(pose):
        data.qpos[:] = 0
        for side in ("left", "right"):
            for k, v in enumerate(pose["joints"][f"{side}_joints"]):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{side}_joint{k+1}")
                data.qpos[model.jnt_qposadr[jid]] = v
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        seg = renderer.render()[:, :, 0]
        out = {}
        for side in ("left", "right"):
            m = np.isin(seg, geom_ids[side]).astype(np.uint8) * 255
            out[side] = cv2.remap(m, maps[0], maps[1], cv2.INTER_NEAREST)
        return out

    def cost_for(side, p):
        x, y, z, yaw = p
        set_base(side, x, y, z, yaw)
        total = 0.0
        for pose in poses:
            rm = render_masks(pose)[side] > 127
            om = pose["obs"][side] > 127
            inter = np.logical_and(rm, om).sum()
            union = np.logical_or(rm, om).sum()
            total += 1.0 - (inter / union if union else 0.0)
        return total / len(poses)

    results = {}
    for side in ("left", "right"):
        sx = -1 if side == "left" else +1
        # Coarse sweep over the two parameters that dominate: how far the base
        # sits from the near edge, and which way it faces.
        best = None
        for yaw_deg in range(30, 151, 10):
            for y in np.arange(-1.00, -0.44, 0.07):
                c = cost_for(side, (sx * 0.25, y, 0.0, math.radians(yaw_deg)))
                if best is None or c < best[0]:
                    best = (c, (sx * 0.25, y, 0.0, math.radians(yaw_deg)))
        print(f"\n{side}: coarse best 1-IoU={best[0]:.4f} at "
              f"y={best[1][1]:+.3f} yaw={math.degrees(best[1][3]):.0f}deg")
        sol = minimize(lambda p: cost_for(side, p), np.array(best[1]),
                       method="Nelder-Mead",
                       options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-5})
        x, y, z, yaw = sol.x
        set_base(side, x, y, z, yaw)
        ious = []
        for pose in poses:
            rm = render_masks(pose)[side] > 127
            om = pose["obs"][side] > 127
            u = np.logical_or(rm, om).sum()
            ious.append(np.logical_and(rm, om).sum() / u if u else 0.0)
        print(f"{side}_mount: pos=({x:+.4f}, {y:+.4f}, {z:+.4f}) "
              f"yaw={math.degrees(yaw):+.2f} deg")
        print(f"  per-pose arm IoU: " + " ".join(f"{v:.3f}" for v in ious)
              + f"   mean={np.mean(ious):.3f}")
        results[side] = {"pos": [float(x), float(y), float(z)],
                         "yaw_deg": float(math.degrees(yaw)),
                         "mean_arm_iou": float(np.mean(ious)),
                         "per_pose_iou": [float(v) for v in ious],
                         "poses": [p["name"] for p in poses]}

    sep = abs(results["left"]["pos"][0] - results["right"]["pos"][0])
    print(f"\nfitted base separation = {sep*100:.1f} cm "
          f"(estimate was 50.0 cm)")
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
