"""RealSense camera discovery: map connected devices to teleop roles.

The relay needs three color streams pinned to fixed roles (top / left wrist
/ right wrist). Device *paths* under ``/dev/v4l/by-id`` are stable across
reboot and replug, but nothing in the path says which physical camera plays
which role — that is a site fact tied to the camera's serial number.

This module owns that mapping. ``SERIAL_TO_ROLE`` below is the committed
source of truth; override it for a different rig by pointing the ``CAM_MAP``
env var at a JSON file of the same ``{"<serial>": "<role>"}`` shape.

Discovery is pure filesystem enumeration (glob ``/dev/v4l/by-id``) + a serial
lookup, so it has no dependency on ``pyrealsense2`` or ``v4l2-ctl`` and works
the same in the relay process and the standalone ``tools/gen_cams_env.py``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

# ── Site config ──────────────────────────────────────────────────────────
# Serial number (as it appears in the by-id device path) → role. Update this
# when you swap a camera, or set CAM_MAP=/path/to/map.json to override without
# editing code. Roles are the canonical short names used everywhere below.
SERIAL_TO_ROLE: dict[str, str] = {
    "254623070531": "top",    # top-down
    "254623070863": "left",   # left wrist
    "254623070417": "right",  # right wrist
}

# Canonical role → (env var the relay reads, camera id in the WS schema, label).
ROLES: dict[str, tuple[str, str, str]] = {
    "top":   ("CAM_TOP",   "top",         "Top"),
    "left":  ("CAM_LEFT",  "left_wrist",  "Left wrist"),
    "right": ("CAM_RIGHT", "right_wrist", "Right wrist"),
}

# The D405 exposes 6 v4l2 nodes per camera; index4 is the YUYV color stream.
# Override with CAM_COLOR_INDEX if a different RealSense model enumerates
# its color node elsewhere.
BY_ID_DIR = Path("/dev/v4l/by-id")
_DEFAULT_COLOR_INDEX = 4

# by-id name looks like:
#   usb-Intel_R__RealSense_..._405_..._254623070531-video-index4
# Capture the serial (digits before -video-index) and the node index.
_BY_ID_RE = re.compile(r"RealSense.*?_(\d+)-video-index(\d+)$")


@dataclass(frozen=True)
class Discovered:
    """One camera found on the bus and matched to a role."""
    role: str
    serial: str
    device: str  # absolute /dev/v4l/by-id path to the color node


@dataclass(frozen=True)
class DiscoveryResult:
    matched: dict[str, Discovered]      # role -> Discovered (only mapped + present)
    unmapped_serials: list[str]         # connected but absent from the map
    missing_roles: dict[str, str]       # role -> serial: mapped but not on the bus


def load_serial_map() -> dict[str, str]:
    """Return the serial→role map, honoring a CAM_MAP JSON override."""
    override = os.environ.get("CAM_MAP", "").strip()
    if not override:
        return dict(SERIAL_TO_ROLE)
    raw = json.loads(Path(override).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"CAM_MAP {override!r} must be a JSON object of serial->role")
    bad = {v for v in raw.values() if v not in ROLES}
    if bad:
        raise ValueError(f"CAM_MAP has unknown roles {sorted(bad)}; valid: {sorted(ROLES)}")
    return {str(k): str(v) for k, v in raw.items()}


def _color_index() -> int:
    raw = os.environ.get("CAM_COLOR_INDEX", "").strip()
    if not raw:
        return _DEFAULT_COLOR_INDEX
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_COLOR_INDEX


def discover(by_id_dir: Path = BY_ID_DIR) -> DiscoveryResult:
    """Enumerate RealSense color nodes and match them to mapped roles."""
    serial_map = load_serial_map()
    color_index = _color_index()

    # serial -> device path for the color node of each connected camera.
    present: dict[str, str] = {}
    if by_id_dir.is_dir():
        for entry in sorted(by_id_dir.iterdir()):
            m = _BY_ID_RE.search(entry.name)
            if not m:
                continue
            serial, node_index = m.group(1), int(m.group(2))
            if node_index == color_index:
                present[serial] = str(entry)

    matched: dict[str, Discovered] = {}
    for serial, device in present.items():
        role = serial_map.get(serial)
        if role is not None:
            matched[role] = Discovered(role=role, serial=serial, device=device)

    unmapped = sorted(s for s in present if s not in serial_map)
    missing = {
        role: serial
        for serial, role in serial_map.items()
        if serial not in present
    }
    return DiscoveryResult(matched=matched, unmapped_serials=unmapped, missing_roles=missing)
