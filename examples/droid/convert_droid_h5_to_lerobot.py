"""
Convert Franka/DROID-style HDF5 episodes into a local LeRobot dataset.

Example:
    uv run examples/droid/convert_droid_h5_to_lerobot.py \
        --raw-dir ../data/panda_raw_episodes_grasp_workpiece_necessary_h5 \
        --repo-id local/pickup_workpiece \
        --task "pick up the workpiece"
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    raw_dir: Path
    repo_id: str = "local/pickup_workpiece"
    task: str = "pick up the workpiece"
    robot_type: str = "franka"
    fps: int = 15
    use_videos: bool = False
    convert_bgr_to_rgb: bool = False
    image_writer_processes: int = max(1, min(8, (os.cpu_count() or 4) // 2))
    image_writer_threads: int = 4
    overwrite: bool = False


def _load_path(file: h5py.File, candidates: tuple[str, ...]) -> np.ndarray:
    for candidate in candidates:
        for key in (candidate, candidate.lstrip("/")):
            try:
                obj = file[key]
                if isinstance(obj, h5py.Group):
                    continue
                return np.asarray(obj)
            except KeyError:
                continue
    raise KeyError(f"Could not find any of these paths: {candidates}")


def _get_group(file: h5py.File, candidates: tuple[str, ...]) -> h5py.Group | None:
    for candidate in candidates:
        for key in (candidate, candidate.lstrip("/")):
            try:
                obj = file[key]
                if isinstance(obj, h5py.Group):
                    return obj
            except KeyError:
                continue
    return None


def _prepare_images(images: np.ndarray, *, convert_bgr_to_rgb: bool) -> np.ndarray:
    images = np.asarray(images)

    if images.ndim != 4:
        raise ValueError(f"Expected image tensor with 4 dims, got shape {images.shape}")

    # Support both (T, H, W, C) and (T, C, H, W).
    if images.shape[-1] not in (1, 3, 4) and images.shape[1] in (1, 3, 4):
        images = np.moveaxis(images, 1, -1)

    if images.shape[-1] == 4:
        images = images[..., :3]
    elif images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)

    if np.issubdtype(images.dtype, np.floating):
        scale = 255.0 if np.max(images) <= 1.0 else 1.0
        images = np.clip(images * scale, 0, 255).astype(np.uint8)
    else:
        images = images.astype(np.uint8, copy=False)

    if convert_bgr_to_rgb:
        images = images[..., ::-1]

    return np.ascontiguousarray(images)


def _prepare_vector(vec: np.ndarray, width: int | None = None) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec[:, None]
    if vec.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {vec.shape}")
    if width is not None and vec.shape[-1] != width:
        raise ValueError(f"Expected width {width}, got shape {vec.shape}")
    return vec


def _load_action(file: h5py.File) -> np.ndarray:
    action_group = _get_group(file, ("/action", "/actions"))
    if action_group is not None:
        if "joint_position" not in action_group or "gripper_position" not in action_group:
            raise KeyError(
                f"Action group must contain joint_position and gripper_position, got keys={list(action_group.keys())}"
            )
        joint_action = _prepare_vector(np.asarray(action_group["joint_position"]), width=7)
        gripper_action = _prepare_vector(np.asarray(action_group["gripper_position"]))
        if gripper_action.shape[-1] != 1:
            gripper_action = gripper_action[:, :1]
        if len(joint_action) != len(gripper_action):
            raise ValueError(
                f"Action lengths do not match: joint_position={len(joint_action)}, gripper_position={len(gripper_action)}"
            )
        return np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32, copy=False)

    action = _prepare_vector(
        _load_path(
            file,
            (
                "/action",
                "/actions",
            ),
        )
    )
    return action.astype(np.float32, copy=False)


def _load_episode(
    episode_path: Path,
    *,
    convert_bgr_to_rgb: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(episode_path, "r") as file:
        base_images = _prepare_images(
            _load_path(
                file,
                (
                    "/observation/exterior_image_1_left",
                    "/observations/exterior_image_1_left",
                    "/observations/images/exterior_image_1_left",
                ),
            ),
            convert_bgr_to_rgb=convert_bgr_to_rgb,
        )
        wrist_images = _prepare_images(
            _load_path(
                file,
                (
                    "/observation/wrist_image",
                    "/observation/wrist_image_left",
                    "/observations/wrist_image",
                    "/observations/images/wrist_image",
                    "/observations/images/wrist_image_left",
                ),
            ),
            convert_bgr_to_rgb=convert_bgr_to_rgb,
        )
        joint_position = _prepare_vector(
            _load_path(
                file,
                (
                    "/observation/joint_position",
                    "/observations/joint_position",
                    "/observations/qpos",
                ),
            ),
            width=7,
        )
        gripper_position = _prepare_vector(
            _load_path(
                file,
                (
                    "/observation/gripper_position",
                    "/observations/gripper_position",
                ),
            )
        )
        action = _load_action(file)

    if gripper_position.shape[-1] != 1:
        gripper_position = gripper_position[:, :1]

    num_frames = len(action)
    for name, value in {
        "base_images": base_images,
        "wrist_images": wrist_images,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }.items():
        if len(value) != num_frames:
            raise ValueError(f"{episode_path} has mismatched lengths: action={num_frames}, {name}={len(value)}")

    return base_images, wrist_images, joint_position, gripper_position, action.astype(np.float32, copy=False)


def _make_dataset(
    repo_id: str,
    *,
    robot_type: str,
    fps: int,
    base_image_shape: tuple[int, int, int],
    wrist_image_shape: tuple[int, int, int],
    action_dim: int,
    use_videos: bool,
    image_writer_processes: int,
    image_writer_threads: int,
) -> LeRobotDataset:
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        features={
            "observation.images.exterior_image_1_left": {
                "dtype": "video" if use_videos else "image",
                "shape": base_image_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist_image_left": {
                "dtype": "video" if use_videos else "image",
                "shape": wrist_image_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "observation.gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["action"],
            },
        },
    )


def main(args: Args) -> None:
    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {args.raw_dir}")

    episode_paths = sorted([*args.raw_dir.glob("episode_*.h5"), *args.raw_dir.glob("episode_*.hdf5")])
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.h5 or episode_*.hdf5 files found under {args.raw_dir}")

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output dataset already exists at {output_path}. Re-run with --overwrite to replace it.")

    sample = _load_episode(episode_paths[0], convert_bgr_to_rgb=args.convert_bgr_to_rgb)
    base_images, wrist_images, _, _, action = sample
    dataset = _make_dataset(
        args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        base_image_shape=tuple(base_images.shape[1:]),
        wrist_image_shape=tuple(wrist_images.shape[1:]),
        action_dim=action.shape[-1],
        use_videos=args.use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    for episode_path in tqdm.tqdm(episode_paths, desc="Converting episodes"):
        base_images, wrist_images, joint_position, gripper_position, action = _load_episode(
            episode_path,
            convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        )

        for i in range(len(action)):
            dataset.add_frame(
                {
                    "task": args.task,
                    "observation.images.exterior_image_1_left": base_images[i],
                    "observation.images.wrist_image_left": wrist_images[i],
                    "observation.joint_position": joint_position[i],
                    "observation.gripper_position": gripper_position[i],
                    "action": action[i],
                }
            )

        dataset.save_episode()

    dataset.consolidate(run_compute_stats=False)
    print(f"Saved LeRobot dataset to: {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
