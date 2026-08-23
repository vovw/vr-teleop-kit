"""Build one dataset from the episodes chosen by tools/sim_select_episodes.py.

Takes the selection JSON (a list of {dataset, episode}) and copies exactly
those episodes, in order, into a fresh LeRobot v3 dataset. Sources must share a
schema and fps; that is checked rather than assumed, because silently merging a
20-dim rot6d EE with a 14-dim quat one would produce a dataset that trains and
is wrong.

Frames come back from LeRobotDataset as float CHW in [0,1] and are converted
to uint8 HWC for the writer, which is exact -- the decoded values are k/255.
The images are re-encoded, so the merged videos are a second AV1 generation off
the originals; at CRF 30 into a 224x168 training resize that is not a concern,
but it is the reason to keep the source datasets rather than re-merging from a
merge.

    python tools/sim_merge_episodes.py --selection sel.json \
        --repo-id Dimios45/spd_stack_cube_sim_clean \
        --root data/spd_stack_cube_sim_clean
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

VECTORS = ("observation.state", "action", "observation.blocks",
           "observation.ee", "action.ee")


def episode_ranges(root: Path) -> dict[int, tuple[int, int]]:
    """episode_index -> [from, to) row range in the dataset's global index."""
    out = {}
    for f in sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"),
                              recursive=True)):
        t = pq.read_table(f)
        for e, a, b in zip(t.column("episode_index").to_pylist(),
                           t.column("dataset_from_index").to_pylist(),
                           t.column("dataset_to_index").to_pylist()):
            out[int(e)] = (int(a), int(b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sel = json.loads(Path(args.selection).read_text())
    if not sel:
        raise SystemExit("selection is empty — nothing to merge")
    out_root = Path(args.root)
    if out_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_root} exists — pass --overwrite to replace")
        shutil.rmtree(out_root)

    srcs = sorted({s["dataset"] for s in sel})
    info = {p: json.loads((Path(p) / "meta" / "info.json").read_text())
            for p in srcs}
    ref = info[srcs[0]]
    for p in srcs[1:]:
        if info[p]["fps"] != ref["fps"]:
            raise SystemExit(f"fps mismatch: {p} is {info[p]['fps']}, "
                             f"{srcs[0]} is {ref['fps']}")
        a = {k: (v["dtype"], tuple(v["shape"])) for k, v in ref["features"].items()}
        b = {k: (v["dtype"], tuple(v["shape"]))
             for k, v in info[p]["features"].items()}
        if a != b:
            raise SystemExit(f"feature schema mismatch between {srcs[0]} and {p}")

    cams = [k.split("observation.images.")[1] for k in ref["features"]
            if k.startswith("observation.images.")]
    features = {k: {"dtype": v["dtype"], "shape": tuple(v["shape"]),
                    "names": v.get("names")}
                for k, v in ref["features"].items()
                if k in VECTORS or k.startswith("observation.images.")}

    print(f"merging {len(sel)} episode(s) from {len(srcs)} dataset(s) "
          f"-> {out_root}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=ref["fps"], root=args.root,
        robot_type=ref.get("robot_type", "bi_yam_sim"), use_videos=bool(cams),
        image_writer_threads=4 * max(len(cams), 1), features=features)

    loaded: dict[str, tuple[LeRobotDataset, dict]] = {}
    t0, done = time.perf_counter(), 0
    total = 0
    for s in sel:
        p = s["dataset"]
        if p not in loaded:
            loaded[p] = (LeRobotDataset(f"merge/{Path(p).name}", root=p),
                         episode_ranges(Path(p)))
        ds, rng = loaded[p]
        a, b = rng[int(s["episode"])]
        for i in range(a, b):
            it = ds[i]
            frame = {k: np.asarray(it[k], dtype=np.float32) for k in VECTORS}
            for cam in cams:
                img = it[f"observation.images.{cam}"]
                # float CHW [0,1] -> uint8 HWC; values are k/255, so exact
                frame[f"observation.images.{cam}"] = (
                    (img.permute(1, 2, 0).numpy() * 255.0).round()
                    .clip(0, 255).astype(np.uint8))
            frame["task"] = it["task"]
            dataset.add_frame(frame)
            done += 1
            if done % 500 == 0:
                r = done / (time.perf_counter() - t0)
                print(f"  {done} frames  ({r:.0f} f/s)", flush=True)
        dataset.save_episode()
        total += b - a
        print(f"  {p} ep{s['episode']} -> ep{dataset.num_episodes-1} "
              f"({b-a} frames)", flush=True)

    dataset.finalize()
    print(f"\n{dataset.num_episodes} episodes, {total} frames "
          f"({total/ref['fps']:.0f} s) at {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
