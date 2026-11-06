import logging
import os
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union
import torch
import torch.distributed as dist
from tqdm import tqdm


import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def split_tokens_evenly(tokens: List[str], num_workers: int) -> List[List[str]]:
    total_count = len(tokens)
    num_workers = min(max(1, num_workers), total_count)
    base_size, extra_count = divmod(total_count, num_workers)
    chunks: List[List[str]] = []
    start_index = 0
    for chunk_index in range(num_workers):
        chunk_size = base_size + (1 if chunk_index < extra_count else 0)
        chunks.append(tokens[start_index : start_index + chunk_size])
        start_index += chunk_size
    return chunks


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    tokens = [t for a in args for t in a["tokens"]]
    all_trajectories = {t: a["trajectories"][t] for a in args for t in a["tokens"]}
    cfg: DictConfig = args[0]["cfg"]


    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    pdm_results: List[pd.DataFrame] = []

    traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )
    
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    for idx, (token) in enumerate(tokens):
        logger.info(
            f"Processing reactive scenario {idx + 1} / {len(tokens)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = all_trajectories[token]

            score_row, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            score_row["valid"] = True
            score_row["log_name"] = metric_cache.log_name
            score_row["frame_type"] = metric_cache.scene_type
            score_row["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row["endpoint_x"] = absolute_endpoint.x
            score_row["endpoint_y"] = absolute_endpoint.y
            score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row = pd.DataFrame([PDMResults.get_empty_results()])
            score_row["valid"] = False

        score_row["token"] = token
        pdm_results.append(score_row)

    return pdm_results


def compute_final_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute final scores after injecting the two-frame extended comfort score
    into the weighted metrics.

    :param pdm_score_df: Original PDM score DataFrame with all tokens
    :return: DataFrame with final score computed
    """
    df = pdm_score_df.reset_index()

    assert (
        not df["two_frame_extended_comfort"].isna().any()
    ), "Found NaN in 'two_frame_extended_comfort'. Please check aggregator completeness."

    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())  # shape: (N, M)
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())  # shape: (N, M)

    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT
    weighted_metrics[:, two_frame_idx] = two_frame_scores

    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)

    assert np.all(total_weight > 0), "Found total_weight == 0 during score computation."

    weighted_metric_scores = weighted_sum / total_weight
    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores

    df.drop(columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"], inplace=True)

    return df


def calculate_weighted_average_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the weighted average for all score columns in the dataframe.
    Automatically normalizes weights if they do not sum to 1.
    """
    score_cols = [c for c in df.columns if c not in {"weight", "token"}]

    if df.empty:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weights = df["weight"].to_numpy()
    scores = df[score_cols].to_numpy()
    total_weight = weights.sum()

    if total_weight == 0:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weighted_avg = (scores * weights[:, None]).sum(axis=0) / total_weight
    return pd.Series(weighted_avg, index=score_cols)


def calculate_individual_mapping_scores(
    pdm_score_df: pd.DataFrame, all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]]
) -> pd.Series:
    """
    Args:
        pdm_score_df: DataFrame containing scores for all tokens
        all_mappings: Two-stage mapping dictionary
    Returns:
        Average score calculated using group multiplication
    """
    all_group_scores = []
    stage1_group_scores = []
    stage2_group_scores = []

    for (orig_token, prev_token), second_stage_pairs in all_mappings.items():

        first_tokens = [pair[0] for pair in second_stage_pairs if len(pair) > 0]
        second_tokens = [pair[1] for pair in second_stage_pairs if len(pair) > 1]

        group1_stage1_df = pdm_score_df[pdm_score_df["token"] == orig_token]
        group1_stage2_df = pdm_score_df[pdm_score_df["token"].isin(first_tokens)]

        group2_stage1_df = pdm_score_df[pdm_score_df["token"] == prev_token]
        group2_stage2_df = pdm_score_df[pdm_score_df["token"].isin(second_tokens)]

        group1_stage1_scores = calculate_weighted_average_score(group1_stage1_df)
        group1_stage2_scores = calculate_weighted_average_score(group1_stage2_df)

        group2_stage1_scores = calculate_weighted_average_score(group2_stage1_df)
        group2_stage2_scores = calculate_weighted_average_score(group2_stage2_df)

        stage1_group_scores.append(group1_stage1_scores)
        stage1_group_scores.append(group2_stage1_scores)

        stage2_group_scores.append(group1_stage2_scores)
        stage2_group_scores.append(group2_stage2_scores)

        group1_scores = group1_stage1_scores * group1_stage2_scores

        group2_scores = group2_stage1_scores * group2_stage2_scores

        avg_scores = (group1_scores + group2_scores) / 2
        all_group_scores.append(avg_scores)

    return (
        pd.DataFrame(all_group_scores).mean(),
        pd.DataFrame(stage1_group_scores).mean(),
        pd.DataFrame(stage2_group_scores).mean(),
    )


def create_scene_aggregators(
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:

    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df["weight"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []

    for (now_frame, previous_frame), second_stage in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            second_stage=second_stage,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        updated_rows = aggregator.aggregate_scores()

        all_updates.append(updated_rows)

    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])

    return full_score_df


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    if rank == 0:
        build_logger(cfg)

    scene_loader = SceneLoader(
        original_sensor_path=None,
        synthetic_sensor_path=None,
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    tokens_to_evaluate = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))

    my_tokens = [tokens_to_evaluate[i] for i in InferenceSampler(len(tokens_to_evaluate))]
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    local_traj = {}
    for token in tqdm(my_tokens, desc=f"[rank{rank}] infer", disable=(rank != 0)):
        scene = scene_loader.get_scene_from_token(token)
        local_traj[token] = agent.compute_trajectory(None, scene) 

    del agent
    torch.cuda.empty_cache()
    
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_traj)

    all_trajectories = {}
    for d in gathered:
        all_trajectories.update(d)
        
    dist.barrier()
    dist.destroy_process_group()

    if rank != 0:
        return None
    
    num_workers = cfg.worker.max_workers
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    
    token_chunks = split_tokens_evenly(tokens_to_evaluate, num_workers)
    
    data_points = [
        {
            "cfg": cfg,
            "tokens": token_list,
            "trajectories": {token: all_trajectories[token] for token in token_list},
        } for token_list in token_chunks
    ]

    logger.info(
    f"PDM scoring: {len(tokens_to_evaluate)} tokens -> "
    f"{len(data_points)} workers, "
    f"~{len(tokens_to_evaluate) // len(data_points)} tokens/worker"
    )

    worker = build_worker(cfg)
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.concat(score_rows)

    try:
        raw_mapping = cfg.train_test_split.reactive_all_mapping
        all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

        for orig_token, prev_token, two_stage_pairs in raw_mapping:
            if prev_token in set(scene_loader.tokens) or orig_token in set(scene_loader.tokens):
                all_mappings[(orig_token, prev_token)] = [tuple(pair) for pair in two_stage_pairs]

        pdm_score_df = create_scene_aggregators(
            all_mappings, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
        )
        pdm_score_df = compute_final_scores(pdm_score_df)
        pseudo_closed_loop_valid = True

    except Exception:
        logger.warning("----------- Failed to calculate pseudo closed-loop weights or comfort:")
        traceback.print_exc()
        pdm_score_df["weight"] = 1.0
        pseudo_closed_loop_valid = False

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    if num_failed_scenarios > 0:
        failed_tokens = pdm_score_df[~pdm_score_df["valid"]]["token"].to_list()
    else:
        failed_tokens = []

    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]

    pcl_group_score, pcl_stage1_score, pcl_stage2_score = calculate_individual_mapping_scores(
        pdm_score_df[score_cols + ["token", "weight"]], all_mappings
    )

    for col in score_cols:
        stage_one_mask = pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL
        stage_two_mask = pdm_score_df["frame_type"] == SceneFrameType.SYNTHETIC

        pdm_score_df.loc[stage_one_mask, f"{col}_stage_one"] = pdm_score_df.loc[stage_one_mask, col]
        pdm_score_df.loc[stage_two_mask, f"{col}_stage_two"] = pdm_score_df.loc[stage_two_mask, col]

    pdm_score_df.drop(columns=score_cols, inplace=True)
    pdm_score_df["score"] = pdm_score_df["score_stage_one"].combine_first(pdm_score_df["score_stage_two"])
    pdm_score_df.drop(columns=["score_stage_one", "score_stage_two"], inplace=True)

    stage1_cols = [f"{col}_stage_one" for col in score_cols if col != "score"]
    stage2_cols = [f"{col}_stage_two" for col in score_cols if col != "score"]
    score_cols = stage1_cols + stage2_cols + ["score"]

    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]

    summary_rows = []

    stage1_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage1_row["token"] = "extended_pdm_score_stage_one"
    stage1_row["valid"] = pseudo_closed_loop_valid
    stage1_row["score"] = pcl_stage1_score.get("score", np.nan)
    for col in pcl_stage1_score.index:
        if col not in ["token", "valid", "score"]:
            stage1_row[f"{col}_stage_one"] = pcl_stage1_score[col]
    summary_rows.append(stage1_row)

    stage2_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage2_row["token"] = "extended_pdm_score_stage_two"
    stage2_row["valid"] = pseudo_closed_loop_valid
    stage2_row["score"] = pcl_stage2_score.get("score", np.nan)
    for col in pcl_stage2_score.index:
        if col not in ["token", "valid", "score"]:
            stage2_row[f"{col}_stage_two"] = pcl_stage2_score[col]
    summary_rows.append(stage2_row)

    combined_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    combined_row["token"] = "extended_pdm_score_combined"
    combined_row["valid"] = pseudo_closed_loop_valid
    combined_row["score"] = pcl_group_score["score"]

    for col in pcl_stage1_score.index:
        if col not in ["token", "valid", "score"]:
            combined_row[f"{col}_stage_one"] = pcl_stage1_score[col]

    for col in pcl_stage2_score.index:
        if col not in ["token", "valid", "score"]:
            combined_row[f"{col}_stage_two"] = pcl_stage2_score[col]
    summary_rows.append(combined_row)

    pdm_score_df = pd.concat([pdm_score_df, pd.DataFrame(summary_rows)], ignore_index=True)

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final extended pdm score of valid results: {pdm_score_df[pdm_score_df["token"] == "extended_pdm_score_combined"]["score"].iloc[0]}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )

    if cfg.verbose:
        logger.info(
            f"""
            Detailed results:
            {pdm_score_df.iloc[-3:].T}
            """
        )
    if num_failed_scenarios > 0:
        logger.info(
            f"""
            List of failed tokens:
            {failed_tokens}
            """
        )


if __name__ == "__main__":

    main()
