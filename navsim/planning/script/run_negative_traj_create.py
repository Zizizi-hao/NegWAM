import os
import uuid
import hydra
import pickle
import logging
import traceback
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Any, Dict, List, Union, Tuple

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.common.dataclasses import SensorConfig
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score_full_v2
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy



devkit_root = os.getenv("NAVSIM_DEVKIT_ROOT")

CONFIG_PATH = f"{devkit_root}/navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_negative_traj"


logger = logging.getLogger(__name__)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    worker = build_worker(cfg)

    vocab = np.load(cfg.vocab_path)

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    if isinstance(scene_filter, tuple) and len(scene_filter) == 1:
        scene_filter = scene_filter[0]
    
    
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=None,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    print(f"Saving results to {cfg.save_dir}")


    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "vocab": vocab
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    new_data_points = []
    for data in data_points:
        for token in data['tokens']:
            new_data_points.append({
                "cfg": cfg,
                "log_file": data['log_file'],
                "token": token,
                "vocab": vocab
            })

    
    score_rows = worker_map(worker, run_pdm_score, new_data_points)
    # score_rows= run_pdm_score(new_data_points)


    final_pdms = {}
    non_negative_traj_token = []
    for tmp in score_rows:
        if not tmp.get("valid", True):
            continue
        final_pdms[tmp['token']] = tmp['score']
        if tmp["non_negative"]:
            non_negative_traj_token.append(tmp['token'])

    with open(f"{cfg.save_dir}/non_negative_traj_token.txt", "w") as f:
        f.write("\n".join(non_negative_traj_token))

    pickle.dump(final_pdms, open(f"{cfg.save_dir}/final_pdms.pkl", "wb"))



def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    # tokens = [t for a in args for t in a["tokens"]]
    tokens = [a["token"] for a in args]
    cfg: DictConfig = args[0]["cfg"]
    vocab_trajectories = args[0]["vocab"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
    )
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=None,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))

    pdm_results: List[Dict[str, Any]] = []

    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token}
        negative_traj_row: Dict[str, Any] = {"token": token}
        scene = scene_loader.get_scene_from_token(token)
        gt = scene.get_future_trajectory(num_trajectory_frames=8)

        try:
            # ======= 1. ADE vocab_trajectories ========
            ade = compute_ade(vocab_trajectories, gt.poses)
            valid_idx = np.where(ade < 1.0)[0]
            if len(valid_idx) > 0:
                vocab_trajectories_sub = vocab_trajectories[valid_idx]
            else:
                # raise ValueError(f"No valid trajectory found for token {token}")
                logger.warning(
                    f"No vocab trajectory within ADE threshold for token {token}, "
                    f"min_ade={ade.min():.3f}m"
                )
                score_row = {"token": token, 
                    "valid": False, 
                    "score": None,
                    "non_negative": True,
                    "skip_reason": "no_ade_match"}
                pdm_results.append(score_row)
                continue

            # ======= 2. pdm_score_full_v2 ========
            metric_cache = metric_cache_loader.get_from_token(token)

            pdm_result,negative_indices = pdm_score_full_v2(
                metric_cache=metric_cache,
                vacab_trajectorys=vocab_trajectories_sub,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )

            # score_row['score'] = pdm_result
            # pdm_score == 0
            pdm_result_negative = {
                key: values[negative_indices]
                for key, values in pdm_result.items()
            }
            score_row['score'] = pdm_result_negative

            negative_traj =  vocab_trajectories_sub[negative_indices]
            negative_traj_row['negative_traj'] = negative_traj

            if negative_traj.shape[0] > 0:
                np.save(f"{cfg.save_dir}/{token}.npy", negative_traj)
                score_row["non_negative"] = False
            else:
                logger.warning(f"No negative trajectory found for token {token}")
                np.save(f"{cfg.save_dir}/{token}.npy", np.array([]))
                score_row["non_negative"] = True

        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row = {
                "valid": False,
                "score": None,
                "non_negative": False,
            }
            #  return None
        
        score_row["token"] = token

        pdm_results.append(score_row)
    
    return pdm_results



def compute_ade(traj_vocab_8, gt_poses):
    """
    traj_vocab_8: (N, 8, 3)
    gt_poses:       (8, 3)
    return: ade (N,)
    """
    if traj_vocab_8.shape[1] != 8:
        sample_indices = np.arange(4, 40, 5)
        traj_vocab_8 = traj_vocab_8[:, sample_indices, :]
    
    traj_vocab_8 = np.asarray(traj_vocab_8)
    gt_poses = np.asarray(gt_poses)

    assert traj_vocab_8.shape[1:] == gt_poses.shape == (8, 3), \
        f"shape mismatch: vocab {traj_vocab_8.shape}, gt {gt_poses.shape}"

    diff = traj_vocab_8[:, :, :2] - gt_poses[None, :, :2]   # (N, 8, 2)
    dist = np.linalg.norm(diff, axis=-1)                    # (N, 8)
    ade = dist.mean(axis=-1)                                  # (N,)
    return ade
    


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(17171)
    # print("Waiting...")
    # debugpy.wait_for_client()
    # print("Debugger attached")
    main()