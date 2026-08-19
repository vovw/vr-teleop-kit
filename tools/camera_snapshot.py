"""Grab a still from each discovered camera, optionally with a crop applied.

Use it to choose and then verify a crop rectangle. The top camera is static so
one look is enough, but the wrist camera moves with the arm: the table horizon
sits at a different row at every pose, so run this at several poses across the
task's workspace before committing to a rectangle. A crop that is clean at the
home pose can be full of room at full extension.

    # what the cameras see right now
    python tools/camera_snapshot.py --out data/camera-check

    # preview a crop (x,y,w,h per camera; 0 for w/h means "to the edge")
    python tools/camera_snapshot.py --out data/camera-check \\
        --crop top=0,0,640,450 --crop right_wrist=0,115,600,365

Writes `<id>.png` and, when a crop is given, `<id>_crop.png` alongside it.
Nothing else may hold the cameras — stop the relay first if it has opened them
for a WebRTC preview.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from vr_teleop_kit.relay.capture import CameraReader, build_camera_specs  # noqa: E402

WARMUP_S = 6.0


def _parse_crop(spec: str) -> tuple[str, tuple[int, int, int, int]]:
    cam, _, rect = spec.partition("=")
    parts = [int(v) for v in rect.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--crop {spec!r} must be <cam>=x,y,w,h")
    return cam.strip(), tuple(parts)  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/camera-check", help="directory for the PNGs")
    ap.add_argument("--crop", action="append", default=[], type=_parse_crop,
                    help="<camera>=x,y,w,h — repeatable; w or h of 0 means to the edge")
    ap.add_argument("--only", action="append", default=[],
                    help="restrict to these camera ids (repeatable)")
    ap.add_argument("--tag", default="",
                    help="suffix for the filenames, e.g. --tag pose2, so snapshots "
                         "from different arm poses don't overwrite each other")
    args = ap.parse_args()

    crops = dict(args.crop)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    specs = [s for s in build_camera_specs() if not args.only or s.id in args.only]
    if not specs:
        raise SystemExit("no cameras discovered (or --only matched none)")

    for spec in specs:
        reader = CameraReader(spec)
        frame = None
        deadline = time.time() + WARMUP_S
        while time.time() < deadline:
            frame = reader.latest()
            if frame is not None:
                break
            time.sleep(0.05)
        reader.stop()
        if frame is None:
            print(f"{spec.id}: no frame in {WARMUP_S:.0f}s — is the relay holding it?")
            continue

        h, w = frame.shape[:2]
        raw_path = out / f"{spec.id}{tag}.png"
        cv2.imwrite(str(raw_path), frame)
        print(f"{spec.id}: {w}x{h} ({spec.label}) -> {raw_path}")

        if spec.id in crops:
            cx, cy, cw, ch = crops[spec.id]
            cw = cw or (w - cx)
            ch = ch or (h - cy)
            if cx < 0 or cy < 0 or cx + cw > w or cy + ch > h:
                print(f"  crop {cx},{cy},{cw},{ch} does not fit in {w}x{h} — skipped")
                continue
            crop_path = out / f"{spec.id}{tag}_crop.png"
            cv2.imwrite(str(crop_path), frame[cy:cy + ch, cx:cx + cw])
            print(f"  crop {cx},{cy},{cw},{ch} -> {crop_path}  ({cw}x{ch})")

    print(f"\nPNGs in {out.resolve()}")


if __name__ == "__main__":
    main()
