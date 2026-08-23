"""Score recorded stacking episodes from the blocks' own poses.

Grasp counts say how hard an operator worked; they do not say whether the task
succeeded. `observation.blocks` (added by examples/record_sim_cubes.py) carries
each block's freejoint pose per frame, so an episode can be graded after the
fact instead of judged by eye.

A stack is called when, at the end of the episode, the blocks' heights sit at
distinct levels one block-height apart and their footprints line up. Tilt is
reported separately -- three blocks can be at the right heights and still be a
leaning tower.

    python tools/sim_score_stack.py --dataset data/sim-stack-35cm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
FIELDS = 7          # x y z qw qx qy qz


def quat_tilt_deg(q: np.ndarray) -> float:
    """How far the block is from resting flat on one of its faces, in degrees.

    Measured as the smallest angle between world +Z and any of the block's six
    face normals -- NOT the angle to its own +Z. For a cube a 90 deg rotation is
    a symmetry: the block looks and sits identically, so reporting the raw +Z
    angle flags a perfectly flat cube as "90 deg, leaning" purely because the
    operator span it while carrying it. That misread made three sound episodes
    look like tip-overs.

    A genuinely toppled or wedged block gives an angle near 45 deg, which no
    face-down orientation can produce.
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    n = np.linalg.norm(R, axis=0)
    if np.any(n < 1e-9):
        return float("nan")
    # columns are the block's own axes in world; +-each is a face normal
    cos = np.abs(R[2, :] / n).max()
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(_REPO_ROOT / "data" / "sim-stack-35cm"))
    ap.add_argument("--block-height", type=float, default=None,
                    help="metres; default: inferred from the resting height at "
                         "the first frame")
    ap.add_argument("--xy-tol", type=float, default=0.020,
                    help="max footprint offset between levels (m)")
    ap.add_argument("--z-tol", type=float, default=0.010,
                    help="max height error per level (m)")
    args = ap.parse_args()

    root = Path(args.dataset)
    d = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    if "observation.blocks" not in d.columns:
        raise SystemExit(
            f"{root} has no 'observation.blocks' — recorded before block poses "
            f"were added. Re-record to score it.")
    info = json.loads((root / "meta" / "info.json").read_text())
    names = info["features"]["observation.blocks"]["names"]
    blocks = [n[:-2] for n in names if n.endswith("_x")]
    B = np.stack(d["observation.blocks"].to_numpy())
    fps = info["fps"]

    print(f"{root}  —  {len(blocks)} blocks: {', '.join(blocks)}")
    for e in sorted(d.episode_index.unique()):
        rows = B[(d.episode_index == e).to_numpy()]
        first, last = rows[0], rows[-1]
        h = args.block_height or float(first[2]) * 2.0     # resting z = h/2
        pos = {b: last[i * FIELDS:i * FIELDS + 3] for i, b in enumerate(blocks)}
        tilt = {b: quat_tilt_deg(last[i * FIELDS + 3:i * FIELDS + 7])
                for i, b in enumerate(blocks)}
        # how far each block moved at all
        moved = {b: float(np.linalg.norm(last[i * FIELDS:i * FIELDS + 3]
                                         - first[i * FIELDS:i * FIELDS + 3]))
                 for i, b in enumerate(blocks)}

        order = sorted(blocks, key=lambda b: pos[b][2])
        zs = np.array([pos[b][2] for b in order])
        levels = np.round((zs - h / 2) / h).astype(int)
        want = np.array([h / 2 + k * h for k in levels])
        z_ok = np.all(np.abs(zs - want) < args.z_tol)
        distinct = len(set(levels.tolist())) == len(blocks)
        xy = np.array([pos[b][:2] for b in order])
        spread = max(np.linalg.norm(xy[i] - xy[j])
                     for i in range(len(xy)) for j in range(len(xy)))
        xy_ok = spread < args.xy_tol
        stacked = bool(z_ok and distinct and xy_ok)
        n_up = int((levels > 0).sum()) + 1 if distinct else int((levels > 0).sum())

        print(f"\n  episode {e}  ({len(rows)} frames, {len(rows)/fps:.1f} s)   "
              f"block height {h*1000:.0f} mm")
        for b in order:
            print(f"    {b:12s} z={pos[b][2]*1000:6.1f} mm  "
                  f"xy=({pos[b][0]*1000:+7.1f},{pos[b][1]*1000:+7.1f})  "
                  f"off-flat={tilt[b]:5.1f}°  moved {moved[b]*1000:6.1f} mm")
        print(f"    levels {levels.tolist()} (distinct={distinct}), "
              f"height error {'ok' if z_ok else 'FAIL'}, "
              f"footprint spread {spread*1000:.1f} mm {'ok' if xy_ok else 'FAIL'}")
        worst_tilt = max(v for v in tilt.values() if not np.isnan(v))
        print(f"    => {'STACK OF ' + str(len(blocks)) if stacked else 'not stacked'}"
              f"   (worst off-flat {worst_tilt:.1f}°"
              + ("  — a block is not sitting flat" if worst_tilt > 10 else "") + ")")


if __name__ == "__main__":
    main()
