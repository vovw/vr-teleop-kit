"""Find the light multiplier that makes the rendered tabletop match the real one.

Geometry being right does not make the render comparable -- if the sim's
tabletop sits 6% brighter than the camera's, every downstream colour statistic
carries that offset. The tabletop is a large flat matte patch, so its median
brightness is a clean handle on total scene illumination.

Only pixels *both* images agree are tabletop are compared, and the real
median is taken over that same region, so specular blooms and the arms do not
skew either side. Brightness in MuJoCo is very nearly linear in light diffuse
below saturation, so a short scan plus interpolation lands within a percent.

    DISPLAY=:0 MUJOCO_GL=glfw python tools/sim_calibrate_lighting.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tools"))

from sim_compare_top import distortion_maps, fill_holes  # noqa: E402
from sim_fit_top_camera import table_mask  # noqa: E402


def render_table_median(scene: Path, intr: dict, real_mask: np.ndarray,
                        maps) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    for gid in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if n.startswith(("left_", "right_")):
            model.geom_rgba[gid, 3] = 0.0
    mujoco.mj_forward(model, data)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    r = mujoco.Renderer(model, intr["height"], intr["width"])
    r.update_scene(data, camera=cam)
    rgb = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
    r.enable_segmentation_rendering()
    r.update_scene(data, camera=cam)
    seg = r.render()[:, :, 0]
    r.disable_segmentation_rendering()
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    mask_i = ((seg == tid).astype(np.uint8)) * 255

    mx, my = maps
    rend = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR)
    rmask = cv2.remap(mask_i, mx, my, cv2.INTER_NEAREST)
    both = (rmask > 127) & (real_mask > 127)
    return np.median(rend[both].reshape(-1, 3), axis=0), both


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=str(_REPO_ROOT / "greenscreen" / "top.png"))
    ap.add_argument("--table-color", default="asshot", choices=("asshot", "neutral"))
    ap.add_argument("--scales", default="0.80,0.88,0.94,1.00,1.06")
    args = ap.parse_args()

    intr = json.loads((_REPO_ROOT / "sim" / "camera_intrinsics.json").read_text())["top"]
    real = cv2.imread(args.real)
    real_mask = fill_holes(table_mask(real))
    maps = distortion_maps(intr, cache=_REPO_ROOT / "sim" / "top_distortion_map.npz")

    scene = _REPO_ROOT / "sim" / "spd_scene.xml"
    scales, meds = [], []
    target = None
    for sc in [float(v) for v in args.scales.split(",")]:
        subprocess.run([sys.executable, str(_REPO_ROOT / "sim" / "build_scene.py"),
                        "--table-color", args.table_color, "--light-scale", str(sc)],
                       check=True, capture_output=True)
        med, both = render_table_median(scene, intr, real_mask, maps)
        if target is None:
            target = np.median(real[both].reshape(-1, 3), axis=0)
            print(f"real tabletop median BGR {target.round(1)} "
                  f"over {int(both.sum())} co-visible px\n")
        lum = float(med.mean())
        scales.append(sc); meds.append(lum)
        print(f"  light_scale={sc:5.3f} -> render median BGR {med.round(1)}  "
              f"lum={lum:6.2f}  (target {target.mean():6.2f})")

    tl = float(target.mean())
    scales, meds = np.array(scales), np.array(meds)
    order = np.argsort(meds)
    best = float(np.interp(tl, meds[order], scales[order]))
    print(f"\ninterpolated light_scale for target lum {tl:.2f}: {best:.4f}")

    subprocess.run([sys.executable, str(_REPO_ROOT / "sim" / "build_scene.py"),
                    "--table-color", args.table_color, "--light-scale", str(best)],
                   check=True, capture_output=True)
    med, _ = render_table_median(scene, intr, real_mask, maps)
    print(f"verify: render median BGR {med.round(1)} vs real {target.round(1)}  "
          f"(lum {med.mean():.2f} vs {tl:.2f}, "
          f"{100*(med.mean()-tl)/tl:+.2f}%)")
    print(f"\nSet LIGHT_SCALE = {best:.4f} in sim/build_scene.py to make it the default.")


if __name__ == "__main__":
    main()
