"""Pull one still frame per episode out of a teleop recording, with joint states.

The wrist-camera extrinsics fit needs (image, gripper pose) pairs, and the
gripper pose comes from forward kinematics on the joint angles recorded in the
same dataset row as the image. That row-level pairing is the whole reason to go
through a LeRobot recording rather than ad-hoc snapshots.

**The calibration pose is at the END of each episode.** The operator drives into
position, holds still for the last few seconds, then presses save. So this takes
the final stationary stretch, not the longest one -- the longest is often at the
start, before the arms moved, which would hand back the *previous* pose while
looking perfectly still.

Frames are read sequentially rather than seeked: mp4 seeks land on keyframes,
which would silently return a frame from a different moment than the joint
state it gets paired with.

    python tools/sim_extract_calib.py --dataset data/wrist-calib

Writes calib/pose_NN/{top,left_wrist,right_wrist}.png + joints.json, and
reports how distinct the poses actually are -- a set that all share a wrist
roll cannot pin the mount transform, however small its residual looks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

CAMERAS = ("top", "left_wrist", "right_wrist")
LEFT_JOINTS = list(range(0, 6))
RIGHT_JOINTS = list(range(7, 13))
STILL_RAD_PER_FRAME = 1e-3
MIN_STILL_FRAMES = 12          # ~0.4 s at 30 fps
TAIL_SEARCH_S = 8.0            # how far back from the end to look


def final_still_window(speed: np.ndarray, fps: int) -> tuple[int, int]:
    """The last run of still frames, searching backwards from the episode end.

    Returns [start, end) in frame indices, or (0, 0) if nothing qualifies.
    Only the tail is considered, so a long motionless stretch at the start of
    the episode cannot win.
    """
    still = speed < STILL_RAD_PER_FRAME
    n = len(still)
    floor = max(0, n - int(TAIL_SEARCH_S * fps))
    i = n
    while i > floor:
        # skip any trailing motion
        while i > floor and not still[i - 1]:
            i -= 1
        end = i
        while i > floor and still[i - 1]:
            i -= 1
        if end - i >= MIN_STILL_FRAMES:
            return i, end
    return 0, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(_REPO_ROOT / "data" / "wrist-calib"))
    ap.add_argument("--out", default=str(_REPO_ROOT / "calib"))
    args = ap.parse_args()

    root = Path(args.dataset)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    data = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    state = np.stack(data["observation.state"].to_numpy())

    picks: list[dict] = []
    for e in sorted(data.episode_index.unique()):
        mask = (data.episode_index == e).to_numpy()
        rows = np.nonzero(mask)[0]
        s = state[mask]
        vl = np.abs(np.diff(s[:, LEFT_JOINTS], axis=0)).max(axis=1)
        vr = np.abs(np.diff(s[:, RIGHT_JOINTS], axis=0)).max(axis=1)
        speed = np.maximum(vl, vr)
        a, b = final_still_window(speed, fps)
        if b - a < MIN_STILL_FRAMES:
            print(f"episode {e}: no still stretch in the last {TAIL_SEARCH_S:.0f}s "
                  f"-- skipped")
            continue
        local = (a + b) // 2
        picks.append({
            "episode": int(e),
            "global_row": int(rows[local]),
            "local_frame": int(local),
            "still_window": [int(a), int(b)],
            "state": [float(v) for v in s[local]],
        })
        print(f"episode {e}: n={len(s)}  final still stretch frames {a}-{b} "
              f"({(b-a)/fps:.1f}s, ending {(len(s)-b)/fps:.1f}s before save)  "
              f"-> picked local {local}")

    if not picks:
        raise SystemExit("no usable frames found")

    wanted = {p["global_row"]: i for i, p in enumerate(picks)}
    out_root = Path(args.out)
    # Decoded with PyAV/libdav1d, not cv2: LeRobot encodes these as AV1 (SVT-AV1)
    # and this OpenCV build has no software AV1 path -- it fails to get a pixel
    # format and returns nothing, silently, for every frame.
    import av

    for cam in CAMERAS:
        vpath = root / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4"
        if not vpath.exists():
            print(f"{cam}: no video at {vpath}")
            continue
        got, idx = 0, 0
        with av.open(str(vpath)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            declared = stream.frames
            for frame in container.decode(stream):
                if idx in wanted:
                    pdir = out_root / f"pose_{wanted[idx]:02d}"
                    pdir.mkdir(parents=True, exist_ok=True)
                    img = frame.to_ndarray(format="bgr24")
                    cv2.imwrite(str(pdir / f"{cam}.png"), img)
                    got += 1
                idx += 1
        if idx != len(data):
            print(f"  {cam}: WARNING decoded {idx} frames but dataset has "
                  f"{len(data)} rows -- row/frame alignment may be off")
        print(f"  {cam}: {got}/{len(picks)} frames written "
              f"(decoded {idx}, container declared {declared})")

    names = info["features"]["observation.state"]["names"]
    for i, p in enumerate(picks):
        pdir = out_root / f"pose_{i:02d}"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "joints.json").write_text(json.dumps({
            "episode": p["episode"],
            "global_row": p["global_row"],
            "local_frame": p["local_frame"],
            "still_window": p["still_window"],
            "fps": fps,
            "state_names": names,
            "state": p["state"],
            "left_joints": p["state"][0:6],
            "left_gripper": p["state"][6],
            "right_joints": p["state"][7:13],
            "right_gripper": p["state"][13],
        }, indent=2) + "\n")

    print("\nchosen poses (rad):")
    for i, p in enumerate(picks):
        print(f"  pose_{i:02d} L=[" + ",".join(f"{v:+.3f}" for v in p["state"][0:6]) + "]")
        print(f"           R=[" + ",".join(f"{v:+.3f}" for v in p["state"][7:13]) + "]")
    arr = np.array([p["state"] for p in picks])
    print("\nspread across poses (max-min, rad):")
    for side, js in (("left", LEFT_JOINTS), ("right", RIGHT_JOINTS)):
        spread = arr[:, js].max(axis=0) - arr[:, js].min(axis=0)
        print(f"  {side:5s} " + " ".join(f"j{k+1}={v:.3f}" for k, v in enumerate(spread)))
        if spread[5] < 0.15:
            print(f"    WARNING {side} joint6 (wrist roll) spans only "
                  f"{spread[5]:.3f} rad -- the mount's rotation about the camera "
                  f"axis will be poorly constrained")
    print(f"\nwrote {out_root}")


if __name__ == "__main__":
    main()
