"""Verify episodes written by `examples/record_yam_bspline_format.py`.

Loads them with the bspline/tidybot2 stack's own `EpisodeReader` — not a
reimplementation — so a pass means their pipeline (`reviewer.py`,
`sort_demos_from_review.py`, `convert_to_robomimic_hdf5.py`) will read the data
too. Then checks the single-YAM schema their `infer_action_meta` dispatches on:

    obs     {arm_pos (3), arm_quat (4, xyzw w>=0), gripper_pos (1), wrist_image}
    action  {arm_pos, arm_quat, gripper_pos}

Usage:
    python tools/check_bspline_episodes.py --input-dir data/demos

Their repo only needs to be importable (numpy + cv2), not installed; point
--bspline-repo at it if it isn't in the default location.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_BSPLINE_REPO = Path.home() / "bspline-policy/real_env/yam_teleop"

# The low-dim keys their `single_yam_rot6d` schema dispatches on. Image keys
# are checked separately: `wrist_image` is what their configs expect, but extra
# feeds (e.g. `top_image` from the dual recorder) are legal — their
# EpisodeReader turns any 3-D observation into its own mp4.
EXPECTED_OBS = {"arm_pos", "arm_quat", "gripper_pos"}
EXPECTED_ACTION = {"arm_pos", "arm_quat", "gripper_pos"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/demos",
                    help="directory of episode dirs (default: data/demos)")
    ap.add_argument("--bspline-repo", default=str(DEFAULT_BSPLINE_REPO),
                    help=f"path to their yam_teleop dir (default: {DEFAULT_BSPLINE_REPO})")
    ap.add_argument("--expect", type=int, default=0,
                    help="fail unless exactly this many episodes are present (0 = any)")
    args = ap.parse_args()

    repo = Path(args.bspline_repo)
    if not (repo / "episode_storage.py").exists():
        raise SystemExit(f"no episode_storage.py under {repo} — pass --bspline-repo")
    sys.path.insert(0, str(repo))
    from episode_storage import EpisodeReader  # their code, unmodified

    root = Path(args.input_dir)
    episode_dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
    if not episode_dirs:
        raise SystemExit(f"no episode directories under {root}")
    if args.expect and len(episode_dirs) != args.expect:
        raise SystemExit(f"expected {args.expect} episodes, found {len(episode_dirs)}")

    print(f"{len(episode_dirs)} episode(s) under {root.resolve()}\n")
    total_frames = 0
    for ep in episode_dirs:
        r = EpisodeReader(ep)  # raises if the layout or pickle is wrong
        n = len(r)
        total_frames += n
        assert n == len(r.timestamps) == len(r.actions), f"{ep.name}: ragged episode"

        obs, act = r.observations[0], r.actions[0]
        image_keys = sorted(k for k in obs
                            if isinstance(obs[k], np.ndarray) and obs[k].ndim == 3)
        assert EXPECTED_OBS <= set(obs), f"{ep.name}: obs keys {sorted(obs)}"
        assert set(obs) - EXPECTED_OBS == set(image_keys), \
            f"{ep.name}: unexpected obs keys {sorted(set(obs) - EXPECTED_OBS - set(image_keys))}"
        assert set(act) == EXPECTED_ACTION, f"{ep.name}: action keys {sorted(act)}"

        for key in image_keys:
            img = np.asarray(obs[key])
            assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8, \
                f"{ep.name}: {key} {img.shape} {img.dtype}"
            # Every step's frame must have come back out of the mp4.
            assert all(o[key] is not None for o in r.observations), \
                f"{ep.name}: {key} did not decode from the mp4 on every step"
        if not image_keys:
            print(f"    WARNING: {ep.name} has no camera feed — unusable for training")
        elif "wrist_image" not in image_keys:
            print(f"    WARNING: {ep.name} has no 'wrist_image' — their task configs "
                  f"expect that key (found {image_keys})")

        for name, d in (("obs", obs), ("action", act)):
            assert np.asarray(d["arm_pos"]).shape == (3,), f"{ep.name}: {name} arm_pos"
            assert np.asarray(d["arm_quat"]).shape == (4,), f"{ep.name}: {name} arm_quat"
            assert np.asarray(d["gripper_pos"]).shape == (1,), f"{ep.name}: {name} gripper_pos"

        quats = np.array([o["arm_quat"] for o in r.observations])
        assert (quats[:, 3] >= 0).all(), f"{ep.name}: quaternion left the w>=0 hemisphere"
        assert np.abs(np.linalg.norm(quats, axis=1) - 1).max() < 1e-9, \
            f"{ep.name}: non-unit quaternion"

        grip = np.array([a["gripper_pos"][0] for a in r.actions])
        assert grip.min() >= 0.0 and grip.max() <= 1.0, f"{ep.name}: gripper out of [0,1]"

        pos = np.array([o["arm_pos"] for o in r.observations])
        travel = float(np.linalg.norm(pos[-1] - pos[0]))
        path = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
        dt = r.timestamps[-1] - r.timestamps[0]
        hz = (n - 1) / dt if dt > 0 else float("nan")
        cams = "+".join(k.replace("_image", "") for k in image_keys) or "no-cam"
        blank = "  (blank frames)" if image_keys and not np.asarray(obs[image_keys[0]]).any() else ""
        print(f"  {ep.name}  {n:4d} frames  {dt:5.1f}s  {hz:5.1f} Hz  "
              f"path {path:.3f} m  net {travel:.3f} m  "
              f"grip {grip.min():.2f}-{grip.max():.2f}  [{cams}]{blank}")

        if path < 1e-3:
            print(f"    WARNING: {ep.name} barely moved — was the clutch held?")

    print(f"\n{total_frames} frames total")
    print("THEIR EpisodeReader ACCEPTS EVERY EPISODE — schema matches single_yam_rot6d")
    print(f"\nnext: python convert_to_robomimic_hdf5.py --input-dir {root.resolve()} "
          f"--output-path demos.hdf5   (run from {repo})")


if __name__ == "__main__":
    main()
