"""Fit the top camera's pose to the real image, instead of guessing it.

We know the tabletop is 1.60 x 1.60 m and the camera sits 0.90 m above it, but
"between the arms, tilted down a bit" is not a pose. The rest of it is
recovered from ``greenscreen/top.png`` itself.

Three features on the table are straight lines in 3D whose world position we
know, so each one constrains the pose:

    far edge   y = +0.80   the table/floor boundary across the top of the frame
    left edge  x = -0.80   runs down-left out of frame at the image edge
    seam       x =  0.00   the butt joint between the two 0.80 m boards

The right edge is deliberately not used: it falls outside the 78.5 deg
horizontal field of view, so it is simply not in the picture. Six line
constraints against five unknowns (lateral, forward, pitch, yaw, roll) leaves
the fit overdetermined, which is what makes the residuals meaningful.

Two things that would quietly bias this if ignored:

* **Lens distortion.** The real edges bow; a MuJoCo render is a perfect
  pinhole. Observed edge pixels are therefore pushed through librealsense's
  own de-projection first, using the same distortion coefficients the factory
  wrote, so both sides of the comparison are ideal-pinhole.
* **MuJoCo's euler composition.** The fitted rotation is checked against the
  ``cam_xmat`` MuJoCo actually builds from the euler we write out, so a
  convention mismatch fails loudly rather than showing up as a 20 px error.

    DISPLAY=:0 MUJOCO_GL=glfw python tools/sim_fit_top_camera.py
    python tools/sim_fit_top_camera.py --free-height   # cross-check the 0.90 m

Writes ``sim/top_camera_pose.json``; re-run ``sim/build_scene.py`` afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

_REPO_ROOT = Path(__file__).resolve().parents[1]

TABLE_HALF = 0.80          # m, half of the 1.60 m tabletop
TOP_CAM_HEIGHT = 0.90      # m, measured above the tabletop

# Column / row windows where each edge is the genuine table boundary. Outside
# these the mask runs into the shelving at the top right, or swaps from being
# the far edge to being the left edge.
FAR_EDGE_COLS = (135, 486)
LEFT_EDGE_ROWS = (108, 335)
RIGHT_EDGE_ROWS = (90, 445)
# Rows where the seam is clear of the grippers intruding at the bottom.
SEAM_ROWS = (105, 430)
# Horizontal band the seam lives in. Without this the matched filter happily
# locks onto the much darker shelf edge at x~600 and the fit silently follows it.
SEAM_COLS = (295, 380)


# ── observation extraction ───────────────────────────────────────────────

def table_mask(img: np.ndarray) -> np.ndarray:
    """Segment the tabletop.

    The melamine reads strongly cyan through the D405's auto-white-balance
    (B well above R) while the wooden floor is warm, which separates them far
    more cleanly than brightness does -- the floor is nearly as bright as the
    table under these lights.

    B > R alone is not enough: the white boxes on the shelving past the table's
    far-right corner also pass it, and they touch the table in the image, so a
    largest-component search swallows them and the whole right edge is lost.
    The tabletop additionally has B ~= G (measured 161/160, 200/196, 210/204)
    where those boxes sit at B >> G (157/137, 140/125) and the floor at B < G,
    so bounding |B - G| separates all three.
    """
    B, G, R = (img[:, :, i].astype(np.int16) for i in range(3))
    m = (((B - R) > 25) & (np.abs(B - G) <= 12) & (G > 60)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))


def far_edge_points(mask: np.ndarray) -> np.ndarray:
    pts = []
    for x in range(*FAR_EDGE_COLS):
        col = np.nonzero(mask[:, x])[0]
        if len(col) and col[0] > 2:
            pts.append((x, col[0]))
    return np.array(pts, float)


def left_edge_points(mask: np.ndarray) -> np.ndarray:
    pts = []
    for y in range(*LEFT_EDGE_ROWS):
        row = np.nonzero(mask[y, :])[0]
        if len(row) and row[0] > 2:
            pts.append((row[0], y))
    return np.array(pts, float)


def right_edge_points(mask: np.ndarray) -> np.ndarray:
    """Rightmost tabletop pixel per row -- the x = +0.80 edge.

    Rows where the table runs off the side of the frame are dropped: there the
    rightmost pixel is the image border, not the table edge.
    """
    w = mask.shape[1]
    pts = []
    for y in range(*RIGHT_EDGE_ROWS):
        row = np.nonzero(mask[y, :])[0]
        if len(row) and row[-1] < w - 3:
            pts.append((row[-1], y))
    return np.array(pts, float)


def seam_points(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Track the dark butt-joint line between the two boards.

    Matched filter per row: a 3 px dark core against the 10 px either side.
    The seam is only a pixel or two of shading, and it is *not* the darkest
    line in the frame -- the shelving on the right and the grippers at the
    bottom are far stronger -- so the search is confined to SEAM_COLS and the
    result is robustly line-fitted to throw out whatever slips through.
    """
    grey = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    grey = grey.astype(np.float32)
    raw: list[tuple[float, float]] = []
    for y in range(*SEAM_ROWS):
        row = grey[y]
        best, best_x = None, None
        for x in range(*SEAM_COLS):
            if not mask[y, x]:
                continue
            core = row[x - 1:x + 2].mean()
            flank = np.concatenate([row[x - 13:x - 3], row[x + 4:x + 14]])
            if len(flank) < 16:
                continue
            resp = core - flank.mean()          # negative = dark line
            if best is None or resp < best:
                best, best_x = resp, x
        if best is None or best > -2.0:
            continue
        a, b, c = row[best_x - 1], row[best_x], row[best_x + 1]
        denom = a - 2 * b + c
        dx = 0.5 * (a - c) / denom if abs(denom) > 1e-6 else 0.0
        raw.append((best_x + float(np.clip(dx, -1.0, 1.0)), y))

    if len(raw) < 20:
        return np.array(raw, float)
    # Robust straight-line fit (the seam is straight), two trim passes.
    pts = np.array(raw, float)
    for _ in range(2):
        A = np.stack([pts[:, 1], np.ones(len(pts))], axis=1)
        coef, *_ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
        resid = np.abs(pts[:, 0] - A @ coef)
        keep = resid < max(2.0, 2.5 * resid.std())
        if keep.sum() < 20:
            break
        pts = pts[keep]
    return pts


# ── geometry ─────────────────────────────────────────────────────────────

def undistort(pts: np.ndarray, intr: dict) -> np.ndarray:
    """Map observed (distorted) pixels to ideal pinhole pixels.

    Uses librealsense's own de-projection so the distortion model matches the
    one the factory calibration was written in (``inverse_brown_conrady``),
    rather than assuming it coincides with OpenCV's Brown-Conrady.
    """
    import pyrealsense2 as rs

    ri = rs.intrinsics()
    ri.width, ri.height = intr["width"], intr["height"]
    ri.fx, ri.fy = intr["fx"], intr["fy"]
    ri.ppx, ri.ppy = intr["ppx"], intr["ppy"]
    ri.model = rs.distortion.inverse_brown_conrady
    ri.coeffs = list(intr["distortion_coeffs"])

    out = np.empty_like(pts)
    for i, (u, v) in enumerate(pts):
        X, Y, Z = rs.rs2_deproject_pixel_to_point(ri, [float(u), float(v)], 1.0)
        out[i] = (intr["fx"] * X / Z + intr["ppx"], intr["fy"] * Y / Z + intr["ppy"])
    return out


def rot_xyz(ex: float, ey: float, ez: float) -> np.ndarray:
    """Rotation for MuJoCo's eulerseq='xyz' (verified against cam_xmat)."""
    cx, sx, cy, sy, cz, sz = (math.cos(ex), math.sin(ex), math.cos(ey),
                              math.sin(ey), math.cos(ez), math.sin(ez))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def project(K: np.ndarray, cam_pos: np.ndarray, R: np.ndarray,
            pts: np.ndarray) -> np.ndarray:
    """World -> ideal pixels, in MuJoCo's camera convention (-Z forward, +Y up)."""
    cam = (pts - cam_pos) @ R
    depth = -cam[:, 2]
    u = K[0, 2] + K[0, 0] * cam[:, 0] / depth
    v = K[1, 2] - K[1, 1] * cam[:, 1] / depth
    return np.stack([u, v], axis=1)


def point_line_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Perpendicular distance from pts to the line through a and b.

    A straight 3D line projects to a straight 2D line under a pinhole, so
    comparing against the line through two projected corners is exact -- no
    need to sample the edge in 3D.
    """
    d = b - a
    n = np.array([-d[1], d[0]])
    n = n / (np.linalg.norm(n) + 1e-12)
    return (pts - a) @ n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(_REPO_ROOT / "greenscreen" / "top.png"))
    ap.add_argument("--free-height", action="store_true",
                    help="also solve for camera height, as a check on the 0.90 m")
    ap.add_argument("--no-seam", action="store_true",
                    help="drop the seam constraint (use if the boards are not equal)")
    ap.add_argument("--fit-features", default="far,left,seam",
                    help="comma-separated features the fit is driven by. Anything "
                         "extracted but not listed is still reported, as a "
                         "held-out check. Default excludes the right edge: it "
                         "runs through shadow and behind the shelving, and the "
                         "morphology needed to segment it there rounds the "
                         "boundary to ~5 px -- good enough to validate a fit, "
                         "not to drive one.")
    ap.add_argument("--out", default=str(_REPO_ROOT / "sim" / "top_camera_pose.json"))
    args = ap.parse_args()

    intr = json.loads((_REPO_ROOT / "sim" / "camera_intrinsics.json").read_text())["top"]
    K = np.array([[intr["fx"], 0, intr["ppx"]],
                  [0, intr["fy"], intr["ppy"]],
                  [0, 0, 1]])

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    mask = table_mask(img)

    obs = {
        "far":  undistort(far_edge_points(mask), intr),
        "left": undistort(left_edge_points(mask), intr),
        "right": undistort(right_edge_points(mask), intr),
    }
    if not args.no_seam:
        sp = seam_points(img, mask)
        if len(sp) > 40:
            obs["seam"] = undistort(sp, intr)
        else:
            print(f"seam: only {len(sp)} points tracked -- dropping the constraint")
    for k, v in obs.items():
        print(f"observed {k:5s}: {len(v):4d} points")

    # World-frame endpoints of each constrained line (z = 0 tabletop).
    H = TABLE_HALF
    lines = {
        "far":  (np.array([-H, +H, 0.0]), np.array([+H, +H, 0.0])),
        "left": (np.array([-H, -H, 0.0]), np.array([-H, +H, 0.0])),
        "right": (np.array([+H, -H, 0.0]), np.array([+H, +H, 0.0])),
        "seam": (np.array([0.0, -H, 0.0]), np.array([0.0, +H, 0.0])),
    }

    def unpack(p):
        x, y, pitch, yaw, roll = p[:5]
        z = p[5] if args.free_height else TOP_CAM_HEIGHT
        return np.array([x, y, z]), rot_xyz(pitch, yaw, roll)

    fit_names = [n.strip() for n in args.fit_features.split(",") if n.strip()]
    fit_names = [n for n in fit_names if n in obs]
    if not fit_names:
        raise SystemExit(f"none of --fit-features present; have {sorted(obs)}")
    held_out = [n for n in obs if n not in fit_names]
    print(f"fitting on: {fit_names}"
          + (f"   held out: {held_out}" if held_out else ""))

    def residuals_for(p, names):
        cam_pos, R = unpack(p)
        out = []
        for name in names:
            a3, b3 = lines[name]
            ab = project(K, cam_pos, R, np.stack([a3, b3]))
            out.append(point_line_dist(obs[name], ab[0], ab[1]))
        return np.concatenate(out)

    def residuals(p):
        return residuals_for(p, fit_names)

    # Seed: centred laterally, at the near edge, aimed at the table centre.
    p0 = [0.0, -TABLE_HALF, math.atan2(TABLE_HALF, TOP_CAM_HEIGHT), 0.0, 0.0]
    if args.free_height:
        p0.append(TOP_CAM_HEIGHT)

    lo = [-0.60, -1.40, math.radians(15), math.radians(-25), math.radians(-25)]
    hi = [+0.60, -0.30, math.radians(75), math.radians(+25), math.radians(+25)]
    if args.free_height:
        lo.append(0.60); hi.append(1.20)
    sol = least_squares(residuals, p0, bounds=(lo, hi),
                        xtol=1e-14, ftol=1e-14, gtol=1e-14)
    cam_pos, R = unpack(sol.x)
    pitch, yaw, roll = sol.x[2:5]

    print(f"\nconverged: {sol.success}  cost={sol.cost:.4f}")
    print(f"camera position   x={cam_pos[0]:+.4f}  y={cam_pos[1]:+.4f}  z={cam_pos[2]:.4f} m")
    print(f"euler (deg)       pitch={math.degrees(pitch):+.3f}  "
          f"yaw={math.degrees(yaw):+.3f}  roll={math.degrees(roll):+.3f}")
    print(f"  -> {math.degrees(pitch) - 90:+.3f} deg from horizontal "
          f"(negative = tilted down)")
    print(f"  -> {cam_pos[1] + TABLE_HALF:+.4f} m from the table's near edge "
          f"(negative = behind it)")

    print(f"\nresiduals (px, perpendicular to each edge):")
    for name in fit_names + held_out:
        seg = residuals_for(sol.x, [name])
        tag = "" if name in fit_names else "  (held out)"
        print(f"  {name:5s} n={len(seg):4d}  rms={np.sqrt((seg**2).mean()):6.3f}  "
              f"mean={seg.mean():+6.3f}  max|.|={np.abs(seg).max():6.3f}{tag}")
    r = residuals(sol.x)
    print(f"  {'FIT':5s} n={len(r):4d}  rms={np.sqrt((r**2).mean()):6.3f}")

    pose = {
        "pos": [float(v) for v in cam_pos],
        "euler_deg": [math.degrees(pitch), math.degrees(yaw), math.degrees(roll)],
        "fitted": True,
        "fit_rms_px": float(np.sqrt((r ** 2).mean())),
        "free_height": bool(args.free_height),
        "source_image": str(Path(args.image).relative_to(_REPO_ROOT)),
        "features": {k: int(len(v)) for k, v in obs.items()},
        "fit_features": fit_names,
        "held_out_features": held_out,
    }
    Path(args.out).write_text(json.dumps(pose, indent=2) + "\n")
    print(f"\nwrote {args.out}")

    # Confirm MuJoCo builds the rotation we think it does from this euler.
    try:
        import mujoco
        xml = f"""<mujoco><compiler angle="radian" eulerseq="xyz"/><worldbody>
        <body name="b" pos="0 0 0" euler="{pitch:.12g} {yaw:.12g} {roll:.12g}">
        <camera name="c"/></body></worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        got = d.cam_xmat[0].reshape(3, 3)
        err = float(np.abs(got - R).max())
        print(f"euler convention check vs MuJoCo cam_xmat: max|diff|={err:.2e}")
        if err > 1e-6:
            print("  WARNING: rotation convention mismatch -- the fit will not "
                  "transfer to the scene correctly")
    except Exception as e:
        print(f"euler convention check skipped: {e}")


if __name__ == "__main__":
    main()
