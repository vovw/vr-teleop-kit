"""Fit each wrist camera's mount on the gripper, from the fingers it can see.

The wrist cameras' *intrinsics* come off the D405 EEPROM and are exact. Their
*extrinsics* -- where the camera sits on the gripper -- were a guess, and a bad
one: the cameras ended up inside the gripper housing, so two of the three vision
inputs rendered gripper internals instead of the scene.

The trick here is that the camera and the fingers are both rigidly attached to
the same gripper body. So the fingers appear at a FIXED place in the wrist image,
and matching rendered fingers against the fingers visible in a real wrist photo
pins the mount directly. That makes this fit independent of:

  * the arm's joint angles,
  * the arm base placement (still only estimated),
  * the table, its position, and its appearance.

All of which the earlier table-edge attempt depended on, which is why it stalled.

The gripper aperture does matter, and is taken from the recorded joint state for
each pose (they were all captured fully open).

    DISPLAY=:0 MUJOCO_GL=glfw python tools/sim_fit_wrist_mount.py --side left

Writes sim/wrist_mount.json plus overlays for eyeballing.
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

DARK_LUM = 70          # fingers are much darker than the melamine
LOWER_FRAME = 0.45     # fingers live below this fraction of the image
MIN_BLOB = 1200
GRIPPER_OPEN_QPOS = 0.0475
GRASP_SITE_Z = 0.1347


def observed_finger_mask(img: np.ndarray) -> np.ndarray:
    """The gripper's own jaws in a real wrist frame: dark, low in the image."""
    h = img.shape[0]
    m = (img.mean(axis=2) < DARK_LUM).astype(np.uint8) * 255
    m[: int(LOWER_FRAME * h), :] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > MIN_BLOB:
            out[lab == i] = 255
    return out


def look_at_quat(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera quaternion (w,x,y,z) aiming at `target` (-Z forward)."""
    fwd = np.asarray(target, float) - np.asarray(cam_pos, float)
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    z = -fwd / n
    u = np.asarray(up, float)
    y = u - np.dot(u, z) * z
    if np.linalg.norm(y) < 1e-6:
        u = np.array([1.0, 0.0, 0.0])
        y = u - np.dot(u, z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def small_rot_quat(rv):
    """Quaternion for a small rotation vector, for perturbing the aim."""
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = rv / th
    return np.array([math.cos(th / 2), *(ax * math.sin(th / 2))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(_REPO_ROOT / "sim" / "spd_scene.xml"))
    ap.add_argument("--calib", default=str(_REPO_ROOT / "calib"))
    ap.add_argument("--side", choices=("left", "right", "both"), default="both")
    ap.add_argument("--out", default=str(_REPO_ROOT / "sim" / "wrist_mount.json"))
    args = ap.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    renderer.enable_segmentation_rendering()

    N = lambda t, n: mujoco.mj_name2id(model, t, n)
    results = {}
    sides = ("left", "right") if args.side == "both" else (args.side,)

    for side in sides:
        cam = N(mujoco.mjtObj.mjOBJ_CAMERA, f"{side}_wrist")
        if cam < 0:
            print(f"{side}: no wrist camera in the scene")
            continue
        # Only the fingertips: in the real frames the visible dark hardware is
        # the two jaws, not the gripper body (which sits behind the lens).
        own = [N(mujoco.mjtObj.mjOBJ_GEOM, f"{side}_{g}")
               for g in ("tip_left", "tip_right")]
        own = [g for g in own if g >= 0]

        poses = []
        for pd in sorted(Path(args.calib).glob("pose_*")):
            f = pd / f"{side}_wrist.png"
            if not f.exists():
                continue
            img = cv2.imread(str(f))
            obs = observed_finger_mask(img)
            if (obs > 127).sum() < 3000:
                print(f"  {pd.name}: only {(obs>127).sum()} finger px, skipped")
                continue
            j = json.loads((pd / "joints.json").read_text())
            poses.append({"name": pd.name, "obs": obs > 127,
                          "grip": float(j[f"{side}_gripper"])})
        if not poses:
            print(f"{side}: no usable calib images")
            continue

        # The fingers are rigid relative to the camera, so they occupy the SAME
        # pixels in every pose. Anything that moves between poses is therefore
        # not a finger -- shadows, dark table, or one frame that came out darker
        # overall. Intersecting the per-pose masks isolates the jaws: on this
        # data only ~20-27% of the union is common, and that common part is
        # exactly the two jaws. Fitting to the union instead was what held the
        # first attempt at IoU 0.15.
        stack = np.stack([p["obs"] for p in poses])
        template = stack.all(axis=0)
        grip = float(np.mean([p["grip"] for p in poses]))
        print(f"  finger template {int(template.sum())} px "
              f"(union was {int(stack.any(axis=0).sum())}), gripper {grip:.3f}")

        jl = N(mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint7")
        jr = N(mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint8")

        def render_mask(grip_recorded: float) -> np.ndarray:
            mujoco.mj_resetData(model, data)
            q = (1.0 - np.clip(grip_recorded, 0, 1)) * GRIPPER_OPEN_QPOS
            data.qpos[model.jnt_qposadr[jl]] = q
            data.qpos[model.jnt_qposadr[jr]] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            seg = renderer.render()[:, :, 0]
            return np.isin(seg, own)

        def pose_quat(p):
            """Free orientation: a rotation vector off the gripper frame.

            Not a look-at. Aiming the camera at the grasp point puts the fingers
            in the middle of the image, but in the real frames they sit along
            the bottom edge -- the camera is pitched to look *past* the jaws at
            the approach region. So orientation is left free and only seeded
            with a pitch.
            """
            return small_rot_quat(p[3:6])

        def cost(p) -> float:
            model.cam_pos[cam] = p[:3]
            model.cam_quat[cam] = pose_quat(p)
            r = render_mask(grip)                 # one render: fingers are static
            inter = np.logical_and(r, template).sum()
            union = np.logical_or(r, template).sum()
            return 1.0 - (inter / union if union else 0.0)

        # Coarse search near the gripper's centreline. The real jaws appear
        # symmetric about the image centre, which only happens if the lens sits
        # close to the axis between them -- an earlier grid over 55-85 mm
        # lateral offsets could not fit the data for that reason. Pitch is
        # swept because it is what slides the fingers from the image centre
        # (pitch 0, looking straight down the approach axis) to the bottom edge.
        best = None
        for lat in (0.0, 0.015, 0.03):
            for fwd in (0.0, 0.025, 0.05):
                for up in (0.0, 0.03, 0.06):
                    for pitch in np.radians((0, 20, 40, 60, 80)):
                        p0 = np.array([lat, fwd, up, pitch, 0.0, 0.0])
                        c = cost(p0)
                        if best is None or c < best[0]:
                            best = (c, p0.copy())
        print(f"  coarse best 1-IoU {best[0]:.4f} at pos "
              f"{np.round(best[1][:3]*1000,1)} mm pitch "
              f"{np.degrees(best[1][3]):.0f}deg")

        sol = minimize(cost, best[1], method="Nelder-Mead",
                       options={"maxiter": 900, "xatol": 1e-4, "fatol": 1e-4})
        pos = sol.x[:3]
        quat = pose_quat(sol.x)
        model.cam_pos[cam] = pos
        model.cam_quat[cam] = quat
        ov = _REPO_ROOT / "sim" / "wrist_fit"
        ov.mkdir(parents=True, exist_ok=True)
        r = render_mask(grip)
        u = np.logical_or(r, template).sum()
        iou = float(np.logical_and(r, template).sum() / u) if u else 0.0
        ious = [iou]
        img = cv2.imread(str(Path(args.calib) / poses[0]["name"] / f"{side}_wrist.png"))
        img[template] = (0.4 * img[template] + np.array([0, 0, 150])).astype(np.uint8)
        img[r] = (0.4 * img[r] + np.array([0, 150, 0])).astype(np.uint8)
        cv2.imwrite(str(ov / f"mount_{side}.png"), img)
        print(f"  fitted pos {np.round(pos*1000,1)} mm  quat {np.round(quat,4)}")
        print(f"  finger-template IoU {iou:.3f}  "
              f"(rendered {int(r.sum())} px vs template {int(template.sum())} px)")
        results[f"{side}_wrist"] = {
            "pos": [float(v) for v in pos],
            "quat_wxyz": [float(v) for v in quat],
            "mean_finger_iou": float(np.mean(ious)),
            "per_pose_iou": [float(v) for v in ious],
            "poses": [p["name"] for p in poses],
        }
    if results:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.out};  overlays in sim/wrist_fit/ "
              f"(red = real fingers, green = rendered)")


if __name__ == "__main__":
    main()
