"""
Extract per-scene training samples from the raw NavSim dataset.

Each scene becomes one ``sample_XXXXXX.pkl`` holding the front-camera frames
(current + future, 2x downsampled), history/future ego trajectories, ego
status, agent boxes, lane polylines around the ego, and camera parameters --
everything NavSimEpisodeDataset and the rollout evaluator need, so training
never touches the raw dataset again.

Scenes are processed with a multiprocessing pool; each worker builds its own
SceneLoader once (see _init_worker).
"""
import argparse
import logging
import multiprocessing as mp
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl

import numpy as np
import yaml
import cv2
from tqdm import tqdm

DRIVEWAM_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
)
if DRIVEWAM_ROOT not in sys.path:
    sys.path.insert(0, DRIVEWAM_ROOT)

# navsim/common/dataclasses.py reads these at import time, so they have to be
# in the environment before the navsim imports below. The defaults assume the
# directory layout from the NavSim setup guide.
os.environ.setdefault("OPENSCENE_DATA_ROOT", "./data/navsim_workspace/dataset")
os.environ.setdefault(
    "NUPLAN_MAPS_ROOT", os.path.join(os.environ["OPENSCENE_DATA_ROOT"], "maps")
)
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import SemanticMapLayer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_WORKER_PROCESSOR = None

CONFIG_PATH = "config/metric_caching"
CONFIG_NAME = "default_feature_caching"


def render_bev_map(map_api, ego_pose, radius=80):
    """Collect lane / lane-connector polylines around the ego, in ego frame."""
    ego_state = StateSE2(ego_pose[0], ego_pose[1], ego_pose[2])

    polyline_layers = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
    ]

    map_object_dict = map_api.get_proximal_map_objects(
        point=ego_state.point,
        radius=radius,
        layers=polyline_layers,
    )

    bev_lanes = []
    for layer in polyline_layers:
        for map_object in map_object_dict[layer]:
            if hasattr(map_object, "baseline_path"):
                linestring = map_object.baseline_path.linestring
                path = np.array(linestring.coords)

                local_x = path[:, 0] - ego_pose[0]
                local_y = path[:, 1] - ego_pose[1]

                cos_h = np.cos(-ego_pose[2])
                sin_h = np.sin(-ego_pose[2])
                rotated_x = local_x * cos_h - local_y * sin_h
                rotated_y = local_x * sin_h + local_y * cos_h

                bev_lanes.append(np.stack([rotated_x, rotated_y], axis=1))

    return bev_lanes


class NavSimDataProcessor:
    """Process NavSim scenes and save one pkl per sample."""

    def __init__(
        self,
        cfg: DictConfig,
        num_history_frames: int = 4,
        num_future_frames: int = 10,
    ):
        self.output_path = Path(cfg.save_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.sensor_blobs_path = Path(cfg.original_sensor_path)
        self.cfg = cfg

        sensor_config = SensorConfig(
            cam_f0=True,
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

        scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)

        self.include_synthetic_scenes = cfg.train_test_split.scene_filter.get("include_synthetic_scenes", False)

        self.scene_loader = SceneLoader(
            original_sensor_path=Path(self.cfg.original_sensor_path),
            synthetic_sensor_path=Path(self.cfg.synthetic_sensor_path) if self.include_synthetic_scenes else None,
            synthetic_scenes_path=Path(self.cfg.synthetic_scenes_path) if self.include_synthetic_scenes else None,
            data_path=Path(cfg.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )

        logger.info("Loaded %d scenes", len(self.scene_loader))

        self.num_history_frames = num_history_frames
        self.num_future_frames = num_future_frames

    def process_scene(self, token: str) -> Dict:
        """Process a single scene token and extract all data."""
        scene = self.scene_loader.get_scene_from_token(token)
        current_idx = self.num_history_frames - 1

        images = []
        for frame_idx in range(current_idx, current_idx + 1 + self.num_future_frames):
            if frame_idx < len(scene.frames):
                frame = scene.frames[frame_idx]
                cam_data = frame.cameras.cam_f0
                images.append(cam_data.image)
        images = np.stack(images, axis=0)

        downsampled_images = []
        for img in images:
            h, w = img.shape[:2]
            img_downsampled = cv2.resize(
                img, (w // 2, h // 2), interpolation=cv2.INTER_AREA
            )
            downsampled_images.append(img_downsampled)
        images = np.stack(downsampled_images, axis=0)

        history_traj = scene.get_history_trajectory()
        history_poses = history_traj.poses

        current_frame = scene.frames[current_idx]
        current_ego = current_frame.ego_status
        ego_status = {
            "pose": current_ego.ego_pose,
            "velocity": current_ego.ego_velocity,
            "acceleration": current_ego.ego_acceleration,
            "driving_command": current_ego.driving_command,
        }


        future_ego_data = []
        if not self.include_synthetic_scenes:
            future_traj = scene.get_future_trajectory()
            future_poses = future_traj.poses

            for i, frame_idx in enumerate(
                range(current_idx + 1, current_idx + 1 + self.num_future_frames)
            ):
                if frame_idx < len(scene.frames):
                    frame = scene.frames[frame_idx]
                    ego = frame.ego_status
                    future_ego_data.append(
                        {
                            "pose": future_poses[i],
                            "velocity": ego.ego_velocity,
                            "acceleration": ego.ego_acceleration,
                        }
                    )
        else:
            future_ego_data =[
                {
                    "pose": np.zeros(3, np.float64),
                    "velocity": np.zeros(2, np.float32),
                    "acceleration": np.zeros(2, np.float32)
                }
                for _ in range(self.num_future_frames)
            ]

        annotations = current_frame.annotations
        agent_boxes = {
            "boxes": annotations.boxes,
            "names": annotations.names,
            "velocity_3d": annotations.velocity_3d,
        }

        cam_f0 = current_frame.cameras.cam_f0
        camera_params = {
            "intrinsics": cam_f0.intrinsics,
            "sensor2lidar_rotation": cam_f0.sensor2lidar_rotation,
            "sensor2lidar_translation": cam_f0.sensor2lidar_translation,
        }

        bev_map = render_bev_map(scene.map_api, ego_status["pose"], radius=80)

        return {
            "images": images,
            "history_poses": history_poses,
            "ego_status": ego_status,
            "future_trajectory": future_ego_data,
            "agent_boxes": agent_boxes,
            "bev_map": bev_map,
            "camera_params": camera_params,
            "map_name": scene.scene_metadata.map_name,
            "metadata": {
                "scene_token": token,
                "log_name": scene.scene_metadata.log_name,
                "map_name": scene.scene_metadata.map_name,
            },
            "prompt": "autonomous driving",
        }

    def process_and_save(self, sample_idx: int, token: str) -> Path:
        """Process one scene and save it to disk."""
        sample = self.process_scene(token)
        sample_file = self.output_path / f"sample_{sample_idx:06d}.pkl"
        with open(sample_file, "wb") as f:
            pickle.dump(sample, f)
        return sample_file


def _init_worker(
    cfg: DictConfig,
    num_history_frames: int,
    num_future_frames: int,
):
    """Create one processor per worker process."""
    global _WORKER_PROCESSOR
    _WORKER_PROCESSOR = NavSimDataProcessor(
        cfg=cfg,
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
    )


def _process_one(task: Tuple[int, str]):
    """Worker entrypoint."""
    global _WORKER_PROCESSOR
    sample_idx, token = task
    try:
        sample_file = _WORKER_PROCESSOR.process_and_save(sample_idx, token)
        return {
            "ok": True,
            "sample_idx": sample_idx,
            "token": token,
            "sample_file": str(sample_file),
        }
    except Exception as exc:
        return {
            "ok": False,
            "sample_idx": sample_idx,
            "token": token,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def process_all_parallel(
    cfg: DictConfig,
    num_history_frames: int = 4,
    num_future_frames: int = 10,

):
    """
    Process all scenes with multiprocessing and save one file per sample.
    """
    processor = NavSimDataProcessor(
        cfg=cfg,
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
    )

    tokens = (
        processor.scene_loader.tokens[:cfg.max_scenes]
        if cfg.max_scenes
        else processor.scene_loader.tokens
    )
    tasks = list(enumerate(tokens))

    if cfg.num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    num_workers = max(1, cfg.num_workers)
    chunksize = max(1, cfg.chunksize)


    # ===========================================================================
    # ===========================================================================
    logger.info(
        "Start multiprocessing with %d workers for %d scenes",
        num_workers,
        len(tasks),
    )

    num_processed = 0
    num_failed = 0
    failed_tokens = []

    if num_workers == 1:
        logger.info("num_workers=1, falling back to single-process execution")
        for sample_idx, token in tqdm(tasks, total=len(tasks), desc="Processing scenes"):
            try:
                processor.process_and_save(sample_idx, token)
                num_processed += 1
            except Exception as exc:
                num_failed += 1
                failed_tokens.append(token)
                logger.error(
                    "Failed to process token %s (sample_%06d): %s",
                    token,
                    sample_idx,
                    str(exc),
                )
                logger.error(traceback.format_exc())
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(
                cfg,
                num_history_frames,
                num_future_frames,
            ),
        ) as pool:

            for result in tqdm(
                pool.imap_unordered(_process_one, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="Processing scenes",
            ):
                if result["ok"]:
                    num_processed += 1
                else:
                    num_failed += 1
                    failed_tokens.append(result["token"])
                    logger.error(
                        "Failed to process token %s (sample_%06d): %s",
                        result["token"],
                        result["sample_idx"],
                        result["error"],
                    )
                    logger.error(result["traceback"])

    logger.info("Finished: %d processed, %d failed", num_processed, num_failed)

    output_dir = Path(cfg.save_dir)
    metadata_file = output_dir / "metadata.txt"
    with open(metadata_file, "w") as f:
        f.write(f"Total samples: {num_processed}\n")
        f.write(f"Failed samples: {num_failed}\n")
        f.write(f"Num history frames: {processor.num_history_frames}\n")
        f.write(f"Num future frames: {processor.num_future_frames}\n")
        f.write(f"Num workers: {num_workers}\n")
        f.write(f"Chunksize: {chunksize}\n")

    if failed_tokens:
        failed_file = output_dir / "failed_tokens.txt"
        with open(failed_file, "w") as f:
            for token in failed_tokens:
                f.write(f"{token}\n")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):

    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    process_all_parallel(cfg=cfg)


if __name__ == "__main__":
    # ===== debug mode =====
    # import debugpy
    # debugpy.listen(17172)
    # print("Waiting...")
    # debugpy.wait_for_client()
    # print("Debugger attached")

    main()
