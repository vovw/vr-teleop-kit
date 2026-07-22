"""v4l2 camera capture — the frame-grabbing half of the camera stack.

Split out from ``relay/server.py`` so both the relay (which fans frames out
to WebRTC tracks) and the dataset recorder (which writes them into a
LeRobotDataset) share one implementation. Discovery/role-mapping lives in
``relay/cameras.py`` and stays dependency-light; this module owns the cv2
capture and therefore depends on ``cv2`` / ``numpy``.

A v4l2 device can only be opened by one process at a time, so exactly one of
{relay, recorder} may own a given camera at once.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vr_teleop_kit.relay.cameras import ROLES, discover
from vr_teleop_kit.log import get_logger

cam_log = get_logger("vr_teleop.camera", "camera")


@dataclass(frozen=True)
class CameraSpec:
    """Camera identity + capture parameters. `id` matches the WS schema
    (top | left_wrist | right_wrist); `label` is the human-facing name;
    `rotate` is 0/90/180/270 degrees applied in the capture thread so
    every consumer sees the corrected frame (no client-side fix-up)."""
    id: str
    label: str
    device: str
    width: int
    height: int
    fps: int
    rotate: int  # 0 | 90 | 180 | 270


# cv2 rotation lookup. 0 → no rotation; other values map to the cv2 constants.
_ROTATE_CODES = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class CameraReader:
    """Background v4l2 grabber. Thread-safe latest-frame slot."""

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._cap = cv2.VideoCapture(spec.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {spec.id} at {spec.device}")
        # MJPG is much cheaper than YUYV for these UVC cams at 640x480@30.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
        self._cap.set(cv2.CAP_PROP_FPS, spec.fps)
        self._thread = threading.Thread(
            target=self._loop, name=f"cam-{spec.id}", daemon=True
        )
        self._thread.start()
        cam_log.info("camera %s opened (%s, %dx%d@%d rotate=%d)",
                    spec.id, spec.device, spec.width, spec.height, spec.fps, spec.rotate)

    def _loop(self) -> None:
        rotate_code = _ROTATE_CODES.get(self.spec.rotate)
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            with self._lock:
                self._frame = frame

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()


# ── Camera registry ──────────────────────────────────────────────────────
# Only cameras whose device path exists are registered. The Quest UI is
# populated from this list (camera_list message on WS open), so missing
# cameras silently disappear instead of erroring at peer-connect time.

def build_camera_specs() -> list[CameraSpec]:
    width = int(os.environ.get("CAM_WIDTH", "640"))
    height = int(os.environ.get("CAM_HEIGHT", "480"))
    fps = int(os.environ.get("CAM_FPS", "30"))

    def _rotate(env_key: str) -> int:
        raw = os.environ.get(env_key, "0").strip() or "0"
        try:
            r = int(raw) % 360
        except ValueError:
            cam_log.warning("%s=%r not an int, defaulting to 0", env_key, raw)
            return 0
        if r not in (0, 90, 180, 270):
            cam_log.warning("%s=%d not in {0,90,180,270}, defaulting to 0", env_key, r)
            return 0
        return r

    # Auto-discover cameras from the bus (serial→role map in cameras.py), then
    # let an explicit env var override the discovered device for any role. This
    # means the relay works with cameras plugged in and no `source cams.env`,
    # while cams.env / manual CAM_* still win when set.
    result = discover()
    for serial in result.unmapped_serials:
        cam_log.warning("camera serial %s connected but not in the role map "
                        "(add it to cameras.SERIAL_TO_ROLE or CAM_MAP)", serial)
    for role, serial in result.missing_roles.items():
        cam_log.info("camera role %s (serial %s) mapped but not on the bus", role, serial)

    specs: list[CameraSpec] = []
    for role, (env_var, cam_id, label) in ROLES.items():
        rotate = _rotate(f"CAM_{role.upper()}_ROTATE")
        device = os.environ.get(env_var)  # explicit override wins
        source = "env"
        if not device:
            found = result.matched.get(role)
            device = found.device if found else None
            source = "discovered"
        if not device:
            cam_log.info("camera %s: no device (env unset, none discovered), skipping", cam_id)
            continue
        if not Path(device).exists():
            cam_log.warning("camera %s: %s (%s) not present, skipping", cam_id, device, source)
            continue
        cam_log.info("camera %s: using %s device %s", cam_id, source, device)
        specs.append(CameraSpec(cam_id, label, device, width, height, fps, rotate))
    return specs
