"""Select the episodes that actually performed the target stacking task.

sim_score_stack.py answers "did three blocks end up stacked". That is weaker
than the task we are training: black cube onto the middle cube, then the right
red cube on top. A run that stacks the same three blocks in a different order,
or that builds the tower on the wrong base, ends in a valid-looking stack while
demonstrating the wrong behaviour -- and imitation learning copies behaviour,
not outcomes.

So an episode passes only when all of the following hold at the last frame:

  * the three blocks sit at three distinct levels one block-height apart, with
    their footprints aligned (the geometric test from sim_score_stack.py),
  * the identity order bottom->top is cube_centre, cube_left, cube_right --
    cube_left is the black block by construction (the recorder assigns colour
    by spawn x, leftmost black), and
  * cube_left settled onto the stack BEFORE cube_right did, i.e. the operator
    worked in the intended order rather than arriving there by luck.

Reads every data shard, so it is correct on datasets grown with --resume.

    python tools/sim_select_episodes.py --dataset data/sim-stack-v5
    python tools/sim_select_episodes.py --dataset A --dataset B --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
FIELDS = 7                     # x y z qw qx qy qz
WANT = ("cube_centre", "cube_left", "cube_right")      # bottom -> top


def quat_tilt_deg(q: np.ndarray) -> float:
    """Smallest angle between world +Z and any face normal (cube symmetry)."""
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    n = np.linalg.norm(R, axis=0)
    if np.any(n < 1e-9):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.abs(R[2, :] / n).max(), -1, 1))))


def settle_frame(z: np.ndarray, h: float) -> int:
    """First frame after which z stays above one block-height, else -1.

    'Above one block height' means the block is resting on another block rather
    than on the table, so this is the moment it joined the stack for good. A
    block that is lifted and put back down never qualifies.
    """
    up = z > h
    if not up[-1]:
        return -1
    # walk back while still up: the settle point is where the final run begins
    i = len(up) - 1
    while i > 0 and up[i - 1]:
        i -= 1
    return int(i)


def score_episode(rows: np.ndarray, blocks: list[str], h: float | None,
                  z_tol: float, xy_tol: float, tilt_tol: float) -> dict:
    first, last = rows[0], rows[-1]
    h = h or float(first[2]) * 2.0
    pos = {b: last[i * FIELDS:i * FIELDS + 3] for i, b in enumerate(blocks)}
    tilt = {b: quat_tilt_deg(last[i * FIELDS + 3:i * FIELDS + 7])
            for i, b in enumerate(blocks)}

    order = sorted(blocks, key=lambda b: pos[b][2])
    zs = np.array([pos[b][2] for b in order])
    levels = np.round((zs - h / 2) / h).astype(int)
    want_z = np.array([h / 2 + k * h for k in levels])
    xy = np.array([pos[b][:2] for b in order])
    spread = max(np.linalg.norm(xy[i] - xy[j])
                 for i in range(len(xy)) for j in range(len(xy)))
    worst_tilt = max(v for v in tilt.values() if not np.isnan(v))

    settle = {b: settle_frame(rows[:, i * FIELDS + 2], h)
              for i, b in enumerate(blocks)}

    reasons = []
    if len(set(levels.tolist())) != len(blocks):
        reasons.append(f"levels not distinct {levels.tolist()}")
    if not np.all(np.abs(zs - want_z) < z_tol):
        reasons.append(f"height error {np.abs(zs-want_z).max()*1000:.0f} mm")
    if spread >= xy_tol:
        reasons.append(f"footprint spread {spread*1000:.0f} mm")
    if worst_tilt > tilt_tol:
        reasons.append(f"off-flat {worst_tilt:.0f}deg")
    if tuple(order) != WANT:
        reasons.append("order " + ">".join(b.replace("cube_", "") for b in order))
    elif settle["cube_left"] < 0 or settle["cube_right"] < 0:
        reasons.append("a block never settled onto the stack")
    elif settle["cube_left"] >= settle["cube_right"]:
        reasons.append("right red placed before black")

    return {"ok": not reasons, "reasons": reasons, "order": list(order),
            "levels": levels.tolist(), "spread_mm": round(spread * 1000, 1),
            "worst_tilt_deg": round(worst_tilt, 1), "block_height_mm": h * 1000,
            "settle": {k: int(v) for k, v in settle.items()}}


def load(root: Path) -> tuple[pd.DataFrame, dict]:
    shards = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not shards:
        raise SystemExit(f"{root}: no data shards")
    d = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
    return d, json.loads((root / "meta" / "info.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--block-height", type=float, default=None)
    ap.add_argument("--z-tol", type=float, default=0.010)
    ap.add_argument("--xy-tol", type=float, default=0.020)
    ap.add_argument("--tilt-tol", type=float, default=10.0)
    ap.add_argument("--json", help="write the selection to this file")
    args = ap.parse_args()

    keep, total = [], 0
    for ds in args.dataset:
        root = Path(ds)
        d, info = load(root)
        names = info["features"]["observation.blocks"]["names"]
        blocks = [n[:-2] for n in names if n.endswith("_x")]
        B = np.stack(d["observation.blocks"].to_numpy()).astype(float)
        fps = info["fps"]
        print(f"\n=== {root}  ({d.episode_index.nunique()} episodes) ===")
        for e in sorted(d.episode_index.unique()):
            rows = B[(d.episode_index == e).to_numpy()]
            r = score_episode(rows, blocks, args.block_height,
                              args.z_tol, args.xy_tol, args.tilt_tol)
            total += 1
            secs = len(rows) / fps
            if r["ok"]:
                keep.append({"dataset": str(root), "episode": int(e),
                             "frames": len(rows), "seconds": round(secs, 1)})
                print(f"  ep{e:<3d} {secs:5.1f}s  KEEP   "
                      f"tilt {r['worst_tilt_deg']:.1f}deg  "
                      f"spread {r['spread_mm']:.0f} mm  "
                      f"black@{r['settle']['cube_left']} "
                      f"red@{r['settle']['cube_right']}")
            else:
                print(f"  ep{e:<3d} {secs:5.1f}s  drop   {'; '.join(r['reasons'])}")

    print(f"\n{len(keep)} of {total} episodes performed the target task "
          f"({100*len(keep)/max(total,1):.0f}%)")
    for k in keep:
        print(f"   {k['dataset']} ep{k['episode']}  {k['seconds']}s")
    if args.json:
        Path(args.json).write_text(json.dumps(keep, indent=2))
        print(f"\nselection written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
