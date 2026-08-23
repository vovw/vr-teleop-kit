"""Verify a recorded sim dataset is complete, self-consistent and LeRobot v3.

Structural checks (shape, dtype, finiteness) are necessary but not sufficient:
they all passed on a dataset whose `observation.ee` was sampled a full control
tick after `observation.state`, so the two representations of the same frame
disagreed by up to a whole unit of gripper travel. The check that caught it was
semantic -- recomputing forward kinematics from the joint observation and
comparing. That check is the reason this file exists, and it runs on every
dataset before the data is trusted.

    python tools/sim_verify_dataset.py --dataset data/sim-stack-v5
    python tools/sim_verify_dataset.py --dataset data/sim-stack-v5 --skip-video

Exits non-zero if any check fails, so it can gate a collection run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "examples"))

ARM_DOFS = 6
HANDS = ("left", "right")
# (joint slice, gripper column, offset into the 20-D EE vector) per arm
ARMS = ((slice(0, 6), 6, 0), (slice(7, 13), 13, 10))
FK_TOL = 1e-5
CAMS = ("top", "left_wrist", "right_wrist")


class Report:
    def __init__(self) -> None:
        self.fails: list[str] = []
        self.warns: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))
        if not ok:
            self.fails.append(label)
        return ok

    def warn(self, label: str, detail: str = "") -> None:
        print(f"  [warn] {label}" + (f"  — {detail}" if detail else ""))
        self.warns.append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--skip-video", action="store_true",
                    help="skip decoding the videos (the slowest check)")
    ap.add_argument("--fk-samples", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.dataset)
    # a resumed dataset spills into file-001, file-002, ... (and eventually
    # chunk-001), so read EVERY data shard, not just the first.
    pqs = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    r = Report()
    if not pqs:
        print(f"{root}: no parquet — nothing recorded")
        return 1
    info = json.loads((root / "meta" / "info.json").read_text())
    d = pd.concat([pd.read_parquet(f) for f in pqs], ignore_index=True)
    if len(pqs) > 1:
        print(f"  (read {len(pqs)} data shards)")
    F = info["features"]
    print(f"\n=== {root} ===")

    # ── format ──────────────────────────────────────────────────────────
    r.check(info["codebase_version"] == "v3.0", "LeRobot codebase_version v3.0",
            info["codebase_version"])
    r.check(len(d) == info["total_frames"],
            "row count matches meta/info.json",
            f"{len(d)} rows vs {info['total_frames']}")
    n_ep = d.episode_index.nunique()
    r.check(n_ep == info["total_episodes"], "episode count matches metadata",
            f"{n_ep} vs {info['total_episodes']}")
    print(f"  {n_ep} episodes, {len(d)} frames, {info['fps']} fps, "
          f"{len(d)/info['fps']:.0f}s total")

    # ── joint space ─────────────────────────────────────────────────────
    for key in ("observation.state", "action"):
        if not r.check(key in F, f"{key} present"):
            continue
        names = F[key]["names"]
        per_arm = [n for n in names if n.startswith("left_")]
        A = np.stack(d[key].to_numpy())
        r.check(A.shape[1] == 14 and len(per_arm) == 7,
                f"{key} is 14-D, 7 DoF per arm",
                f"{A.shape[1]}-D, left {len(per_arm)}")
        r.check(bool(np.isfinite(A).all()), f"{key} all finite")

    # ── end-effector space ──────────────────────────────────────────────
    for key in ("observation.ee", "action.ee"):
        if not r.check(key in F, f"{key} present"):
            continue
        A = np.stack(d[key].to_numpy())
        r.check(A.shape[1] == 20, f"{key} is 20-D", f"{A.shape[1]}-D")
        r.check(bool(np.isfinite(A).all()), f"{key} all finite")
        norms = [np.linalg.norm(A[:, o + s:o + s + 3], axis=1)
                 for o in (0, 10) for s in (3, 6)]
        worst = max(float(np.abs(n - 1.0).max()) for n in norms)
        r.check(worst < 2e-3, f"{key} rot6d columns unit-norm",
                f"max deviation {worst:.2e}")

    # ── the semantic check: EE must agree with the joints ───────────────
    if all(k in d.columns for k in
           ("observation.state", "observation.ee", "action", "action.ee")):
        from record_bi_ee_only import ROTATION_ENCODINGS, _ee_vector
        from vr_teleop_kit.ik.decoupled_ik import DecoupledIKSolver
        enc, _ = ROTATION_ENCODINGS["rot6d"]
        fk = DecoupledIKSolver()
        S = np.stack(d["observation.state"].to_numpy())
        EE = np.stack(d["observation.ee"].to_numpy())
        A = np.stack(d["action"].to_numpy())
        EA = np.stack(d["action.ee"].to_numpy())
        oerr = aerr = 0.0
        for i in np.linspace(0, len(S) - 1, min(args.fk_samples, len(S))).astype(int):
            for js, gs, eo in ARMS:
                ref = np.asarray(_ee_vector(fk, np.asarray(S[i, js], float),
                                            float(S[i, gs]), enc))
                oerr = max(oerr, float(np.abs(ref - EE[i, eo:eo + 10]).max()))
                refa = np.asarray(_ee_vector(fk, np.asarray(A[i, js], float),
                                             float(A[i, gs]), enc))
                aerr = max(aerr, float(np.abs(refa - EA[i, eo:eo + 10]).max()))
        r.check(oerr < FK_TOL, "observation.ee == FK(observation.state)",
                f"max err {oerr:.2e}")
        r.check(aerr < FK_TOL, "action.ee == FK(action)", f"max err {aerr:.2e}")

    # ── blocks ──────────────────────────────────────────────────────────
    if r.check("observation.blocks" in F, "observation.blocks present"):
        B = np.stack(d["observation.blocks"].to_numpy())
        r.check(B.shape[1] % 7 == 0, "blocks are 7 values each (pos + quat)",
                f"{B.shape[1]}-D = {B.shape[1]//7} blocks")
        r.check(bool(np.isfinite(B).all()), "observation.blocks all finite")
        qn = [np.linalg.norm(B[:, k * 7 + 3:k * 7 + 7], axis=1)
              for k in range(B.shape[1] // 7)]
        worst = max(float(np.abs(n - 1.0).max()) for n in qn)
        r.check(worst < 1e-3, "block quaternions normalised",
                f"max deviation {worst:.2e}")
        # every episode should start from a different randomised layout
        firsts = np.stack([B[(d.episode_index == e).to_numpy()][0]
                           for e in sorted(d.episode_index.unique())])
        xs = firsts[:, [k * 7 for k in range(B.shape[1] // 7)]]
        spread = float(xs.max() - xs.min())
        r.check(spread > 0.02, "spawn positions randomised across episodes",
                f"x spread {spread*1000:.0f} mm")

    # ── gripper conventions ─────────────────────────────────────────────
    S = np.stack(d["observation.state"].to_numpy())
    g = np.concatenate([S[:, 6], S[:, 13]])
    r.check(g.min() >= -1e-6 and g.max() <= 1 + 1e-6,
            "gripper in 0..1 (0=open, 1=closed)",
            f"[{g.min():.2f}, {g.max():.2f}]")

    # ── videos ──────────────────────────────────────────────────────────
    if args.skip_video:
        r.warn("video frame alignment not checked (--skip-video)")
    else:
        import av
        for cam in CAMS:
            key = f"observation.images.{cam}"
            if key not in F:
                r.check(False, f"{cam} video feature present")
                continue
            vids = sorted((root / "videos" / key).glob("chunk-*/file-*.mp4"))
            if not r.check(bool(vids), f"{cam} video file exists",
                           f"{len(vids)} file(s)" if vids
                           else str(root / "videos" / key)):
                continue
            n = 0
            for p in vids:
                with av.open(str(p)) as c:
                    n += sum(1 for _ in c.decode(c.streams.video[0]))
            r.check(n == len(d), f"{cam} has one frame per row",
                    f"{n} frames vs {len(d)} rows")

    # ── per-episode sanity: did anything actually happen? ───────────────
    print("  per-episode:")
    for e in sorted(d.episode_index.unique()):
        s = S[(d.episode_index == e).to_numpy()]
        secs = len(s) / info["fps"]
        grips = [int((((s[:, gi] > 0.5)[:-1] == False) &
                      ((s[:, gi] > 0.5)[1:])).sum()) for gi in (6, 13)]
        flag = ""
        if secs < 5:
            flag = "  <- suspiciously short, likely an accidental save"
        elif sum(grips) == 0:
            flag = "  <- no grasp attempted"
        print(f"    ep{e}: {len(s):5d}f {secs:5.1f}s  grasps L{grips[0]}/R{grips[1]}{flag}")
        if secs < 5 or sum(grips) == 0:
            r.warn(f"episode {e} looks unused", f"{secs:.1f}s, {sum(grips)} grasps")

    print()
    if r.fails:
        print(f"FAILED {len(r.fails)} check(s): {', '.join(r.fails)}")
        return 1
    print(f"ALL CHECKS PASSED" + (f"  ({len(r.warns)} warning(s))" if r.warns else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
