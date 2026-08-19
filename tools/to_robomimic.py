"""Convert either recorded dataset into a robomimic HDF5 for bspline/diffusion training.

Two input formats, two action spaces, one tool — so the crop, the resize and
the RGB convention are defined in exactly one place for both:

  --from bspline   their episode dirs (`data.pkl` + `<key>.mp4`)
                   -> obs {arm_pos(3), arm_quat(4), gripper_pos(1), images}
                      actions (N, 7) = [pos(3), rotvec(3), gripper(1)]
                   Their dataset class turns the rotvec into rotation_6d, so
                   the trained action is 10-dim: their `single_yam_rot6d`.

  --from lerobot   a LeRobot v3 dataset with 7-dim joint state/action
                   -> obs {joint_pos(7), images}
                      actions (N, 7) = [joint1..6 (rad), gripper]
                   Their `single_yam_joint` format: no rotation conversion and
                   no IK at deploy.

With no --crop the bspline path is byte-identical to their
`convert_to_robomimic_hdf5.py`; this tool adds cropping and the joint-space
input.

Cropping happens here rather than at record time on purpose: the recorded mp4s
stay full-resolution, so a crop can be retuned and the HDF5 rebuilt without
re-recording. Whatever rectangle you pick MUST also be applied to the
observation at deployment, or the policy sees a distribution it never trained
on. Crops are stored in the HDF5 attrs so the choice travels with the data.

    python tools/to_robomimic.py --from bspline \\
        --input-dir data/demos-ee-pick-duster \\
        --output-path ~/bspline-policy/data/yam_ee.hdf5 \\
        --crop top_image=42,28,598,414

    python tools/to_robomimic.py --from lerobot \\
        --repo-id Dimios45/yam-pick-duster --root data/lerobot-pick-duster \\
        --output-path ~/bspline-policy/data/yam_joint.hdf5 \\
        --crop top_image=42,28,598,414
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

DEFAULT_BSPLINE_REPO = Path.home() / "bspline-policy/real_env/yam_teleop"
# Their constants.POLICY_IMAGE_WIDTH / HEIGHT — what the policy server feeds
# the network at inference, so the dataset must match.
POLICY_IMAGE_SIZE = 84

# LeRobot camera key -> the obs key their configs expect.
LEROBOT_IMAGE_KEYS = {
    "observation.images.right_wrist": "wrist_image",
    "observation.images.left_wrist": "wrist_image",
    "observation.images.top": "top_image",
}


def _parse_crop(spec: str):
    cam, _, rect = spec.partition("=")
    parts = [int(v) for v in rect.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--crop {spec!r} must be <image_key>=x,y,w,h")
    return cam.strip(), tuple(parts)


def _prepare(img: np.ndarray, key: str, crops: dict, size: int) -> np.ndarray:
    """Crop (optional) then resize to the policy's input size. RGB uint8 in and out."""
    if key in crops:
        x, y, w, h = crops[key]
        ih, iw = img.shape[:2]
        w = w or (iw - x)
        h = h or (ih - y)
        if x < 0 or y < 0 or x + w > iw or y + h > ih:
            raise SystemExit(f"crop {x},{y},{w},{h} for {key} does not fit in {iw}x{ih}")
        img = img[y:y + h, x:x + w]
    return cv2.resize(img, (size, size))


def _quat_xyzw_to_rotvec(quat_xyzw: np.ndarray) -> np.ndarray:
    """Axis-angle from an xyzw quaternion — matches scipy's `as_rotvec`, which
    is what their converter uses, without taking a scipy dependency."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / np.linalg.norm(q)
    if q[3] < 0.0:  # shortest rotation
        q = -q
    angle = 2.0 * np.arccos(np.clip(q[3], -1.0, 1.0))
    s = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if s < 1e-12:  # tiny angle: axis is ill-conditioned, series expansion instead
        return 2.0 * q[:3]
    return (angle / s) * q[:3]


def convert_bspline(args, crops: dict) -> tuple[int, int, dict]:
    sys.path.insert(0, str(Path(args.bspline_repo)))
    try:
        from episode_storage import EpisodeReader  # their code, unmodified
    except ImportError as e:
        raise SystemExit(f"cannot import their episode_storage from {args.bspline_repo}: {e}")

    root = Path(args.input_dir)
    episode_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.max_episodes:
        episode_dirs = episode_dirs[:args.max_episodes]
    if not episode_dirs:
        raise SystemExit(f"no episode dirs under {root}")

    n_frames = 0
    obs_keys: dict = {}
    with h5py.File(args.output_path, "w") as f:
        data = f.create_group("data")
        for idx, ep in enumerate(episode_dirs):
            r = EpisodeReader(ep)
            obs: dict[str, list] = {}
            for o in r.observations:
                for k, v in o.items():
                    v = np.asarray(v)
                    if v.ndim == 3:
                        v = _prepare(v, k, crops, args.image_size)
                    obs.setdefault(k, []).append(v)
            actions = [np.concatenate((
                np.asarray(a["arm_pos"], dtype=np.float64),
                _quat_xyzw_to_rotvec(a["arm_quat"]),
                np.asarray(a["gripper_pos"], dtype=np.float64),
            )) for a in r.actions]

            g = data.create_group(f"demo_{idx}")
            for k, v in obs.items():
                g.create_dataset(f"obs/{k}", data=np.array(v))
            g.create_dataset("actions", data=np.array(actions))
            n_frames += len(r)
            obs_keys = {k: np.array(v).shape[1:] for k, v in obs.items()}
            print(f"  demo_{idx:<3d} {len(r):4d} frames  {ep.name}")
        _stamp(f, args, crops, "bspline", "single_yam_rot6d")
    return len(episode_dirs), n_frames, obs_keys


def convert_lerobot(args, crops: dict) -> tuple[int, int, dict]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=args.root)
    state_dim = ds.meta.features["observation.state"]["shape"][0]
    if state_dim != 7:
        raise SystemExit(
            f"expected a 7-dim joint state (joint1..6 + gripper), got {state_dim}. "
            "This path is for single-arm joint-space datasets.")

    episode_index = np.array(ds.hf_dataset["episode_index"])
    n_eps = ds.num_episodes if not args.max_episodes else min(args.max_episodes, ds.num_episodes)

    n_frames = 0
    obs_keys: dict = {}
    with h5py.File(args.output_path, "w") as f:
        data = f.create_group("data")
        for idx in range(n_eps):
            rows = np.flatnonzero(episode_index == idx)
            joint_pos, actions = [], []
            images: dict[str, list] = {}
            for row in rows:
                item = ds[int(row)]
                joint_pos.append(item["observation.state"].numpy().astype(np.float64))
                actions.append(item["action"].numpy().astype(np.float64))
                for cam_key, out_key in LEROBOT_IMAGE_KEYS.items():
                    if cam_key not in item:
                        continue
                    # LeRobot hands back CHW float32 in [0, 1], RGB.
                    img = (item[cam_key].numpy().transpose(1, 2, 0) * 255.0)
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    images.setdefault(out_key, []).append(
                        _prepare(img, out_key, crops, args.image_size))

            g = data.create_group(f"demo_{idx}")
            g.create_dataset("obs/joint_pos", data=np.array(joint_pos))
            for k, v in images.items():
                g.create_dataset(f"obs/{k}", data=np.array(v))
            g.create_dataset("actions", data=np.array(actions))
            n_frames += len(rows)
            obs_keys = {"joint_pos": (7,), **{k: np.array(v).shape[1:] for k, v in images.items()}}
            print(f"  demo_{idx:<3d} {len(rows):4d} frames")
        _stamp(f, args, crops, "lerobot", "single_yam_joint")
    return n_eps, n_frames, obs_keys


def _stamp(f, args, crops: dict, source: str, action_format: str) -> None:
    """Record how this HDF5 was built, so the deployment side can reproduce the
    exact image pipeline instead of relying on someone's memory."""
    f.attrs["source_format"] = source
    f.attrs["action_format"] = action_format
    f.attrs["image_size"] = args.image_size
    f.attrs["crops"] = json.dumps({k: list(v) for k, v in crops.items()})
    f.attrs["gripper_convention"] = "0=open, 1=closed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="source", choices=("bspline", "lerobot"), required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--crop", action="append", default=[], type=_parse_crop,
                    help="<image_key>=x,y,w,h, e.g. top_image=42,28,598,414 (repeatable)")
    ap.add_argument("--image-size", type=int, default=POLICY_IMAGE_SIZE,
                    help=f"square size fed to the policy (default: {POLICY_IMAGE_SIZE})")
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    # bspline source
    ap.add_argument("--input-dir", help="[--from bspline] directory of episode dirs")
    ap.add_argument("--bspline-repo", default=str(DEFAULT_BSPLINE_REPO))
    # lerobot source
    ap.add_argument("--repo-id", help="[--from lerobot] dataset repo id")
    ap.add_argument("--root", help="[--from lerobot] local dataset root")
    args = ap.parse_args()

    crops = dict(args.crop)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    if args.source == "bspline":
        if not args.input_dir:
            raise SystemExit("--from bspline needs --input-dir")
        n_eps, n_frames, obs_keys = convert_bspline(args, crops)
    else:
        if not args.repo_id:
            raise SystemExit("--from lerobot needs --repo-id (and usually --root)")
        n_eps, n_frames, obs_keys = convert_lerobot(args, crops)

    size_mb = Path(args.output_path).stat().st_size / 1e6
    print(f"\n{n_eps} demos, {n_frames} frames -> {args.output_path} ({size_mb:.0f} MB)")
    print("obs keys:", {k: tuple(v) for k, v in obs_keys.items()})
    print("crops   :", {k: list(v) for k, v in crops.items()} or "none")
    print("\nshape_meta for the task yaml:")
    for k, shape in obs_keys.items():
        if len(shape) == 3:
            print(f"    {k}:\n      shape: [3, {shape[0]}, {shape[1]}]\n      type: rgb")
        else:
            print(f"    {k}:\n      shape: [{shape[0]}]")
    print("  action:")
    print(f"    shape: [{10 if args.source == 'bspline' else 7}]")


if __name__ == "__main__":
    main()
