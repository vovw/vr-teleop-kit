"""Record the factory intrinsics of the three RealSense D405s to JSON.

The relay captures colour through plain v4l2 (``cv2.VideoCapture``), which
tells us nothing about the lens. The intrinsics live in the camera's own
calibration EEPROM, so we read them once with librealsense and freeze them
into ``sim/camera_intrinsics.json``. Everything downstream (the MuJoCo scene,
the pose fit, the render comparison) reads that file instead of re-opening
the cameras, so the sim stays reproducible with the rig powered off.

Role mapping is the subtle part. ``relay/cameras.py`` keys roles off the
serial in the ``/dev/v4l/by-id`` path, but librealsense reports a *different*
number as ``serial_number`` (the module serial). The by-id serial is what
librealsense calls ``asic_serial_number``, so that is what we match on --
and it keeps working for a camera whose USB descriptor comes up without a
serial at all, which is how the top camera currently enumerates.

    python tools/sim_dump_intrinsics.py                 # all roles, 640x480
    python tools/sim_dump_intrinsics.py --width 1280 --height 720

Nothing else may hold the cameras: stop the relay first.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from vr_teleop_kit.relay.cameras import ROLES, load_serial_map  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=str(_REPO_ROOT / "sim" / "camera_intrinsics.json"))
    args = ap.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise SystemExit(
            "pyrealsense2 is required to read factory intrinsics "
            "(uv pip install pyrealsense2)"
        ) from e

    serial_to_role = load_serial_map()
    devices = list(rs.context().query_devices())
    if not devices:
        raise SystemExit("no RealSense devices found")

    out: dict[str, dict] = {}
    for dev in devices:
        # asic_serial_number is the number that shows up in the by-id path and
        # therefore the one relay/cameras.py maps to a role.
        asic = dev.get_info(rs.camera_info.asic_serial_number)
        role = serial_to_role.get(asic)
        if role is None:
            print(f"  skip asic={asic}: not in the role map")
            continue
        cam_id = ROLES[role][1]

        intr = None
        for sensor in dev.query_sensors():
            for prof in sensor.get_stream_profiles():
                if not prof.is_video_stream_profile():
                    continue
                vp = prof.as_video_stream_profile()
                if (vp.stream_type() != rs.stream.color
                        or vp.width() != args.width
                        or vp.height() != args.height
                        or vp.fps() != args.fps):
                    continue
                intr = vp.get_intrinsics()
                break
            if intr is not None:
                break
        if intr is None:
            print(f"  {cam_id}: no colour profile at "
                  f"{args.width}x{args.height}@{args.fps}")
            continue

        hfov = 2 * math.degrees(math.atan(intr.width / (2 * intr.fx)))
        vfov = 2 * math.degrees(math.atan(intr.height / (2 * intr.fy)))
        out[cam_id] = {
            "role": role,
            "camera_id": cam_id,
            "asic_serial": asic,
            "module_serial": dev.get_info(rs.camera_info.serial_number),
            "firmware": dev.get_info(rs.camera_info.firmware_version),
            "width": intr.width,
            "height": intr.height,
            "fps": args.fps,
            "fx": intr.fx, "fy": intr.fy,
            "ppx": intr.ppx, "ppy": intr.ppy,
            "distortion_model": str(intr.model),
            "distortion_coeffs": list(intr.coeffs),
            "hfov_deg": hfov,
            "vfov_deg": vfov,
        }
        print(f"  {cam_id:12s} asic={asic}  fx={intr.fx:.4f} fy={intr.fy:.4f} "
              f"ppx={intr.ppx:.4f} ppy={intr.ppy:.4f}  "
              f"HFOV={hfov:.3f}° VFOV={vfov:.3f}°")

    missing = [ROLES[r][1] for r in ROLES if ROLES[r][1] not in out]
    if missing:
        print(f"missing roles: {missing}")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
