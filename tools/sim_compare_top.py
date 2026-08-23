"""Render the sim's top camera and compare it against the real frame.

This is the check that the pixel match actually holds end to end. Two details
decide whether the comparison is honest:

* **Distortion goes on the render, not off the photo.** A MuJoCo render is a
  perfect pinhole; the D405 frame is not. Undistorting the photo would resample
  and soften the very edges we are measuring, so instead the render is pushed
  through the real lens model (librealsense's own de-projection, with the
  factory coefficients) and compared against the untouched frame. The same
  remap is what you would apply to make sim renders match real camera output
  for training.
* **The arms are hidden by default.** ``greenscreen/top.png`` was captured
  without synchronised joint states, so the arm pose in the photo is unknown
  and a rendered arm would just add noise. Geometry being validated here is
  the table and the camera; pass ``--show-arms`` to draw them anyway.

    DISPLAY=:0 MUJOCO_GL=glfw python tools/sim_compare_top.py

Writes real/render/overlay/blend/diff PNGs and prints IoU plus boundary error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tools"))

from sim_fit_top_camera import table_mask  # noqa: E402


def distortion_maps(intr: dict, cache: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Remap that turns an ideal pinhole render into what this lens would see.

    For every output (distorted) pixel we ask librealsense where that pixel's
    ray points, then re-project the ray through the *same* intrinsics with no
    distortion -- giving the source coordinate in the pinhole render.
    """
    if cache and cache.exists():
        z = np.load(cache)
        return z["map_x"], z["map_y"]

    import pyrealsense2 as rs

    ri = rs.intrinsics()
    ri.width, ri.height = intr["width"], intr["height"]
    ri.fx, ri.fy = intr["fx"], intr["fy"]
    ri.ppx, ri.ppy = intr["ppx"], intr["ppy"]
    ri.model = rs.distortion.inverse_brown_conrady
    ri.coeffs = list(intr["distortion_coeffs"])

    W, H = intr["width"], intr["height"]
    map_x = np.empty((H, W), np.float32)
    map_y = np.empty((H, W), np.float32)
    for v in range(H):
        for u in range(W):
            X, Y, Z = rs.rs2_deproject_pixel_to_point(ri, [float(u), float(v)], 1.0)
            map_x[v, u] = intr["fx"] * X / Z + intr["ppx"]
            map_y[v, u] = intr["fy"] * Y / Z + intr["ppy"]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, map_x=map_x, map_y=map_y)
    return map_x, map_y


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Close interior holes so arm occlusions do not count against the IoU."""
    # The padding ring must start as *non-table*, so that every non-table
    # region touching the image border connects to it and gets flooded away.
    # What survives inside is a genuine hole. (Seeding a zero ring instead
    # makes the fill a no-op and silently returns the whole frame.)
    h, w = mask.shape
    pad = np.full((h + 2, w + 2), 255, np.uint8)
    pad[1:-1, 1:-1] = np.where(mask > 127, 0, 255).astype(np.uint8)
    cv2.floodFill(pad, None, (0, 0), 0)
    holes = pad[1:-1, 1:-1] > 0
    out = mask.copy()
    out[holes] = 255
    return out


# Rows above the grippers/arms intruding from the bottom of the top-camera
# frame. Below this the real mask has arm-shaped bites the arm-free render has
# no counterpart for, and any boundary distance there measures the arms.
BOUNDARY_ROWS = (0, 330)


def boundary_error(real: np.ndarray, rend: np.ndarray,
                   rows: tuple[int, int] = BOUNDARY_ROWS) -> dict:
    """Directional distances between the two table outlines, in pixels.

    Only boundary pixels inside `rows` are *scored*, but they may match to a
    boundary pixel anywhere -- so the window excludes the arm silhouettes
    without inventing a false edge at the cut line.
    """
    def edge(m):
        return cv2.Canny(m, 50, 150) > 0

    er, es = edge(real), edge(rend)
    if not er.any() or not es.any():
        return {}
    d_to_real = cv2.distanceTransform(255 - (er * 255).astype(np.uint8), cv2.DIST_L2, 5)
    d_to_rend = cv2.distanceTransform(255 - (es * 255).astype(np.uint8), cv2.DIST_L2, 5)
    sel = np.zeros_like(er)
    sel[rows[0]:rows[1], :] = True
    out = {}
    for name, src, dist in (("render_to_real", es & sel, d_to_real),
                            ("real_to_render", er & sel, d_to_rend)):
        if not src.any():
            continue
        d = dist[src]
        out[name] = (float(np.median(d)), float(d.mean()),
                     float(np.percentile(d, 90)), int(src.sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(_REPO_ROOT / "sim" / "spd_scene.xml"))
    ap.add_argument("--real", default=str(_REPO_ROOT / "greenscreen" / "top.png"))
    ap.add_argument("--out", default=str(_REPO_ROOT / "sim" / "compare"))
    ap.add_argument("--show-arms", action="store_true")
    ap.add_argument("--keyframe", default=None,
                    help="load this keyframe's qpos before rendering (e.g. ready)")
    args = ap.parse_args()

    import mujoco

    intr = json.loads((_REPO_ROOT / "sim" / "camera_intrinsics.json").read_text())["top"]
    W, H = intr["width"], intr["height"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    real = cv2.imread(args.real)
    if real is None:
        raise SystemExit(f"cannot read {args.real}")

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    if not args.show_arms:
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith(("left_", "right_")):
                model.geom_rgba[gid, 3] = 0.0
    if args.keyframe:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
        if kid < 0:
            raise SystemExit(f"no keyframe named {args.keyframe!r}")
        mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)

    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    renderer = mujoco.Renderer(model, H, W)

    renderer.update_scene(data, camera=cam)
    ideal = renderer.render()                       # RGB, ideal pinhole

    # Segmentation pass gives an exact tabletop mask, no colour thresholding.
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam)
    seg = renderer.render()[:, :, 0]
    renderer.disable_segmentation_rendering()
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    rend_mask_ideal = ((seg == table_gid).astype(np.uint8)) * 255

    map_x, map_y = distortion_maps(
        intr, cache=_REPO_ROOT / "sim" / "top_distortion_map.npz")
    ideal_bgr = cv2.cvtColor(ideal, cv2.COLOR_RGB2BGR)
    rend = cv2.remap(ideal_bgr, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rend_mask = cv2.remap(rend_mask_ideal, map_x, map_y, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    real_mask = fill_holes(table_mask(real))
    rm = rend_mask > 127
    tm = real_mask > 127
    inter, union = (rm & tm).sum(), (rm | tm).sum()
    iou = inter / union if union else float("nan")
    be = boundary_error(real_mask, rend_mask)

    print(f"tabletop mask IoU (real vs render): {iou:.4f}")
    print(f"boundary outline agreement (rows {BOUNDARY_ROWS[0]}-{BOUNDARY_ROWS[1]}, "
          f"clear of the arms):")
    for k, (med, mean, p90, n) in be.items():
        print(f"  {k:15s} n={n:5d}  median {med:5.2f}  mean {mean:5.2f}  "
              f"p90 {p90:6.2f} px")
    print(f"real mask {tm.sum()} px, render mask {rm.sum()} px "
          f"({100*(rm.sum()-tm.sum())/tm.sum():+.2f}% area)")

    # Colour check over the region both agree is tabletop.
    both = rm & tm
    if both.any():
        cr = real[both].reshape(-1, 3).astype(float)
        cs = rend[both].reshape(-1, 3).astype(float)
        print(f"tabletop colour  real median BGR {np.median(cr,0).round(1)}  "
              f"render median BGR {np.median(cs,0).round(1)}")

    cv2.imwrite(str(out / "real.png"), real)
    cv2.imwrite(str(out / "render_ideal.png"), ideal_bgr)
    cv2.imwrite(str(out / "render_distorted.png"), rend)

    # Rendered table outline drawn over the real frame -- the clearest read on
    # whether the geometry lines up.
    overlay = real.copy()
    cnts, _ = cv2.findContours(rend_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 2)
    cnts_r, _ = cv2.findContours(real_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, cnts_r, -1, (0, 255, 0), 1)
    cv2.imwrite(str(out / "overlay_edges.png"), overlay)

    cv2.imwrite(str(out / "blend.png"), cv2.addWeighted(real, 0.5, rend, 0.5, 0))
    cv2.imwrite(str(out / "diff.png"),
                cv2.applyColorMap(cv2.convertScaleAbs(
                    cv2.absdiff(real, rend).max(axis=2)), cv2.COLORMAP_INFERNO))
    sbs = np.hstack([real, rend])
    cv2.putText(sbs, "REAL", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(sbs, "SIM", (W + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(out / "side_by_side.png"), sbs)
    print(f"\nwrote {out}/ (real, render_ideal, render_distorted, "
          f"overlay_edges, blend, diff, side_by_side)")


if __name__ == "__main__":
    main()
