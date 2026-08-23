"""Fit each wrist camera's mount transform on its gripper, from a teleop recording.

The wrist intrinsics come off the D405 EEPROM and are exact. What is unknown is
where each camera sits on the gripper -- 6 numbers per side. Solving that needs
the gripper's world pose when each frame was taken, which is why this consumes
`calib/pose_NN/` (frames paired with joint states by sim_extract_calib.py).

The mount is a *single* unknown shared across every pose. Per pose the gripper's
world pose comes from forward kinematics on the recorded joints, so
camera pose = gripper pose o mount; project a known table feature through that
and compare against the same feature found in the image. Five diverse poses
over-determine six unknowns.

**The feature is the table's far edge, labelled.** Several things that looked
better are not:

* The table *silhouette* is unusable. From 20-40 cm above a 1.60 m table only
  the far edge is in frame, and the mask's remaining boundary traces the gripper
  jaws and the clutter on the table -- both touch the image border, so they
  survive hole-filling and then dominate an ICP fit.
* Unlabelled ICP against all four edges is worse than one labelled line: three
  edges project off-frame, and standing in a constant for them flattens the
  objective until every seed reaches the same cost.
* The board seam is a strong line in some poses and a cable on a dark object in
  others, so it is not trusted here. Stage two below brings it in only where it
  agrees with the far-edge solution.

Seeds are the 24 axis-aligned orientations, not random ones: a camera bracket is
machined, so the lens is square to the gripper axes give or take a few degrees.

    DISPLAY=:0 MUJOCO_GL=glfw python tools/sim_fit_wrist_cameras.py

Writes sim/wrist_camera_poses.json and overlays in sim/wrist_fit/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[1]

TABLE_HALF = 0.80
FAR_EDGE_Y = +TABLE_HALF     # world line the far edge lies on
SEAM_X = 0.0                 # world line the board seam lies on
MIN_EDGE_PTS = 30
MAX_PTS = 400                # subsample cap per feature, for speed
RANSAC_ITERS = 900
RANSAC_THR = 3.0


# ── segmentation ─────────────────────────────────────────────────────────

def table_masks(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (core, filled, clutter) tabletop masks.

    `core` is the raw bright/unsaturated region -- objects on the table are
    holes in it. `filled` closes those holes. `clutter` is what the filling
    added, dilated, so ridge searches can avoid the edges of objects.

    The recording's white balance is near-neutral, unlike the cyan-cast
    greenscreen stills, so brightness + low saturation is the discriminator
    here rather than a colour cast.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    raw = ((V > 120) & (S < 70)).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(raw, 8)
    if n < 2:
        z = np.zeros_like(raw)
        return raw, raw, z
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    core = (lab == big).astype(np.uint8) * 255
    h, w = core.shape
    pad = np.full((h + 2, w + 2), 255, np.uint8)
    pad[1:-1, 1:-1] = np.where(core > 127, 0, 255).astype(np.uint8)
    cv2.floodFill(pad, None, (0, 0), 0)
    filled = core.copy()
    filled[pad[1:-1, 1:-1] > 0] = 255
    clutter = cv2.dilate(((filled > 127).astype(np.uint8) * 255) - core,
                         np.ones((9, 9), np.uint8))
    return core, filled, clutter


def ransac_line(pts: np.ndarray, rng, thr: float = RANSAC_THR,
                iters: int = RANSAC_ITERS, min_span: float = 30.0):
    """Dominant straight line in a point set; returns its inliers.

    RANSAC rather than a plain fit because the topmost-table-pixel trace is
    only *partly* the far edge -- where an object sits near the far edge, the
    topmost table pixel is the object's lower boundary instead. Those are
    outliers to the real edge, and a least-squares fit would split the
    difference.
    """
    if len(pts) < MIN_EDGE_PTS:
        return None
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        a, b = pts[i], pts[j]
        d = b - a
        L = float(np.hypot(*d))
        if L < min_span:
            continue
        nrm = np.array([-d[1], d[0]]) / L
        inl = np.abs((pts - a) @ nrm) < thr
        c = int(inl.sum())
        if best is None or c > best[0]:
            best = (c, inl)
    if best is None or best[0] < MIN_EDGE_PTS:
        return None
    return pts[best[1]]


def far_edge_points(core: np.ndarray) -> np.ndarray | None:
    """Topmost tabletop pixel per column, reduced to its dominant line."""
    h, w = core.shape
    pts = []
    for x in range(w):
        col = np.nonzero(core[:, x])[0]
        if len(col) and 3 < col[0] < h - 4:
            pts.append((x, col[0]))
    if len(pts) < MIN_EDGE_PTS:
        return None
    return np.array(pts, float)


def seam_candidates(img: np.ndarray, core: np.ndarray,
                    clutter: np.ndarray) -> np.ndarray:
    """Dark thin-line pixels inside the tabletop, away from clutter edges."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    g = g.astype(np.float32)
    ridge = cv2.GaussianBlur(g, (0, 0), 5.0) - g
    # Search inside `core`, not `filled`. `filled` closes interior holes, but an
    # object touching the image border is not a hole -- it stays outside `core`
    # and is invisible to the clutter mask. Searching `filled` therefore sweeps
    # over such objects, and a cable lying on one gives a stronger dark-line
    # response than the seam itself.
    inner = cv2.erode(core, np.ones((21, 21), np.uint8))
    ok = (inner > 127) & (clutter == 0)
    if not ok.any():
        return np.empty((0, 2))
    thr = max(float(np.percentile(ridge[ok], 99.3)), 2.5)
    ys, xs = np.nonzero((ridge > thr) & ok)
    return np.stack([xs, ys], axis=1).astype(float)


# ── geometry ─────────────────────────────────────────────────────────────

def undistort(pts: np.ndarray, intr: dict) -> np.ndarray:
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


def project(K, cam_pos, R, pts):
    cam = (pts - cam_pos) @ R
    depth = -cam[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    u = K[0, 2] + K[0, 0] * cam[:, 0] / safe
    v = K[1, 2] - K[1, 1] * cam[:, 1] / safe
    return np.stack([u, v], axis=1), depth


def axis_aligned_rotations() -> list[np.ndarray]:
    """The 24 proper rotations with entries in {0, +-1}."""
    out, eye = [], np.eye(3)
    for i in range(3):
        for si in (1, -1):
            for j in range(3):
                if j == i:
                    continue
                for sj in (1, -1):
                    a, b = si * eye[i], sj * eye[j]
                    R = np.stack([a, b, np.cross(a, b)], axis=1)
                    if abs(np.linalg.det(R) - 1.0) < 1e-9:
                        out.append(R)
    return out


def line_residual(K, cam_pos, R, world_a, world_b, obs_pts, miss=120.0):
    """Perpendicular distance from obs_pts to the projected world line a-b."""
    uv, depth = project(K, cam_pos, R, np.stack([world_a, world_b]))
    if depth[0] <= 0.05 or depth[1] <= 0.05:
        return np.full(len(obs_pts), miss)
    d = uv[1] - uv[0]
    L = float(np.hypot(*d))
    if L < 1e-6:
        return np.full(len(obs_pts), miss)
    nrm = np.array([-d[1], d[0]]) / L
    return (obs_pts - uv[0]) @ nrm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=str(_REPO_ROOT / "calib"))
    ap.add_argument("--scene", default=str(_REPO_ROOT / "sim" / "spd_scene.xml"))
    ap.add_argument("--seam-tol", type=float, default=6.0,
                    help="stage 2 accepts a detected seam only if it already "
                         "agrees with the far-edge fit to within this many px")
    ap.add_argument("--out", default=str(_REPO_ROOT / "sim" / "wrist_camera_poses.json"))
    args = ap.parse_args()

    import mujoco

    intr_all = json.loads((_REPO_ROOT / "sim" / "camera_intrinsics.json").read_text())
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    pose_dirs = sorted(Path(args.calib).glob("pose_*"))
    if not pose_dirs:
        raise SystemExit(f"no pose_* dirs under {args.calib}")

    rng = np.random.default_rng(0)
    results = {}
    H = TABLE_HALF
    FAR_A, FAR_B = np.array([-H, FAR_EDGE_Y, 0.0]), np.array([+H, FAR_EDGE_Y, 0.0])
    SEAM_A, SEAM_B = np.array([SEAM_X, -H, 0.0]), np.array([SEAM_X, +H, 0.0])

    for side in ("left", "right"):
        cam_id = f"{side}_wrist"
        intr = intr_all[cam_id]
        K = np.array([[intr["fx"], 0, intr["ppx"]],
                      [0, intr["fy"], intr["ppy"]], [0, 0, 1]])
        gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_gripper")

        obs = []
        for pd in pose_dirs:
            img_p = pd / f"{cam_id}.png"
            if not img_p.exists():
                continue
            j = json.loads((pd / "joints.json").read_text())
            q = j[f"{side}_joints"]
            data.qpos[:] = 0
            for k, v in enumerate(q):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{side}_joint{k+1}")
                data.qpos[model.jnt_qposadr[jid]] = v
            mujoco.mj_forward(model, data)

            img = cv2.imread(str(img_p))
            core, filled, clutter = table_masks(img)
            raw = far_edge_points(core)
            fe = ransac_line(raw, rng) if raw is not None else None
            if fe is None:
                print(f"  {pd.name} {cam_id}: no far edge -- skipped")
                continue
            if len(fe) > MAX_PTS:
                fe = fe[rng.choice(len(fe), MAX_PTS, replace=False)]
            seam_raw = seam_candidates(img, core, clutter)
            seam = ransac_line(seam_raw, rng, thr=2.5) if len(seam_raw) else None
            if seam is not None and len(seam) > MAX_PTS:
                seam = seam[rng.choice(len(seam), MAX_PTS, replace=False)]
            obs.append({
                "name": pd.name,
                "G_pos": data.xpos[gb].copy(),
                "G_mat": data.xmat[gb].reshape(3, 3).copy(),
                "far": undistort(fe, intr),
                "seam": undistort(seam, intr) if seam is not None else None,
                "q": q, "shape": img.shape[:2],
            })
            print(f"  {pd.name} {cam_id}: far edge {len(fe)} px, "
                  f"seam {0 if seam is None else len(seam)} px")

        if len(obs) < 3:
            print(f"{side}: only {len(obs)} usable poses -- need at least 3")
            continue

        def make_res(use_seam: dict[str, bool] | None):
            def res(p):
                t = p[:3]
                R_m = Rotation.from_rotvec(p[3:6]).as_matrix()
                out = []
                for o in obs:
                    cam_pos = o["G_pos"] + o["G_mat"] @ t
                    R = o["G_mat"] @ R_m
                    out.append(line_residual(K, cam_pos, R, FAR_A, FAR_B, o["far"]))
                    if use_seam and use_seam.get(o["name"]) and o["seam"] is not None:
                        out.append(line_residual(K, cam_pos, R, SEAM_A, SEAM_B,
                                                 o["seam"]))
                return np.concatenate(out)
            return res

        lo = np.array([-0.15, -0.15, -0.15, -np.inf, -np.inf, -np.inf])
        hi = np.array([+0.15, +0.15, +0.15, +np.inf, +np.inf, +np.inf])

        # stage 1: far edge only
        res1 = make_res(None)
        cands = []
        for R0 in axis_aligned_rotations():
            r0 = Rotation.from_matrix(R0).as_rotvec()
            for t0 in ([0, 0, 0], [0.05, 0, -0.03], [-0.05, 0, -0.03],
                       [0, 0.05, -0.03], [0, -0.05, -0.03]):
                p0 = np.clip(np.concatenate([t0, r0]), lo, hi)
                try:
                    s = least_squares(res1, p0, bounds=(lo, hi), loss="soft_l1",
                                      f_scale=5.0, max_nfev=150,
                                      xtol=1e-10, ftol=1e-10)
                except Exception as e:
                    print(f"    seed failed: {type(e).__name__}: {e}")
                    continue
                cands.append((s.cost, s.x))
        if not cands:
            print(f"{side}: all seeds failed")
            continue
        cands.sort(key=lambda c: c[0])
        print(f"  {side}: stage-1 best costs {[round(c[0], 1) for c in cands[:5]]}")
        sol = least_squares(res1, cands[0][1], bounds=(lo, hi), loss="soft_l1",
                            f_scale=3.0, xtol=1e-14, ftol=1e-14)

        # stage 2: fold in each seam that already agrees with stage 1
        keep = {}
        t, R_m = sol.x[:3], Rotation.from_rotvec(sol.x[3:6]).as_matrix()
        for o in obs:
            if o["seam"] is None:
                continue
            cam_pos = o["G_pos"] + o["G_mat"] @ t
            R = o["G_mat"] @ R_m
            d = line_residual(K, cam_pos, R, SEAM_A, SEAM_B, o["seam"])
            med = float(np.median(np.abs(d)))
            keep[o["name"]] = med < args.seam_tol
            print(f"    seam check {o['name']}: median |d|={med:6.2f} px -> "
                  f"{'accepted' if keep[o['name']] else 'rejected'}")
        if any(keep.values()):
            sol = least_squares(make_res(keep), sol.x, bounds=(lo, hi),
                                loss="soft_l1", f_scale=3.0, xtol=1e-14, ftol=1e-14)

        # report per-feature residuals
        t, R_m = sol.x[:3], Rotation.from_rotvec(sol.x[3:6]).as_matrix()
        rot = Rotation.from_rotvec(sol.x[3:6])
        print(f"\n{cam_id}:")
        print(f"  mount translation (gripper frame) = "
              f"{np.round(t, 4).tolist()} m   |t| = {np.linalg.norm(t)*100:.1f} cm")
        print(f"  mount rotation  (euler xyz, deg)  = "
              f"{np.round(rot.as_euler('xyz', degrees=True), 2).tolist()}")
        all_fe = []
        for o in obs:
            cam_pos = o["G_pos"] + o["G_mat"] @ t
            R = o["G_mat"] @ R_m
            d = line_residual(K, cam_pos, R, FAR_A, FAR_B, o["far"])
            all_fe.append(d)
            print(f"    {o['name']}: far-edge rms {np.sqrt((d**2).mean()):6.2f} px "
                  f"(n={len(d)})")
        fe = np.concatenate(all_fe)
        print(f"  far-edge overall rms = {np.sqrt((fe**2).mean()):.2f} px "
              f"over {len(fe)} px, {len(obs)} poses")

        ov = _REPO_ROOT / "sim" / "wrist_fit"
        ov.mkdir(parents=True, exist_ok=True)
        for o in obs:
            img = cv2.imread(str(Path(args.calib) / o["name"] / f"{cam_id}.png"))
            cam_pos = o["G_pos"] + o["G_mat"] @ t
            R = o["G_mat"] @ R_m
            for a, b, col in ((FAR_A, FAR_B, (0, 0, 255)),
                              (SEAM_A, SEAM_B, (0, 200, 255))):
                uv, dep = project(K, cam_pos, R, np.stack([a, b]))
                if dep[0] > 0.05 and dep[1] > 0.05:
                    p1 = tuple(np.round(uv[0]).astype(int))
                    p2 = tuple(np.round(uv[1]).astype(int))
                    cv2.line(img, p1, p2, col, 2)
            for u, v in o["far"]:
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    cv2.circle(img, (int(u), int(v)), 1, (0, 255, 0), -1)
            cv2.imwrite(str(ov / f"{o['name']}_{cam_id}.png"), img)
        print(f"  overlays -> {ov}")

        results[cam_id] = {
            "mount_pos": [float(v) for v in t],
            "mount_euler_deg": [float(v) for v in rot.as_euler("xyz", degrees=True)],
            "mount_quat_wxyz": [float(v) for v in np.roll(rot.as_quat(), 1)],
            "far_edge_rms_px": float(np.sqrt((fe ** 2).mean())),
            "poses_used": [o["name"] for o in obs],
            "seams_accepted": [k for k, v in keep.items() if v],
        }

    if results:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
