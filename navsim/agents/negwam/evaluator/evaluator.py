import argparse
import dataclasses
import lzma
import os
import pickle
import sys
import tempfile
import torch.multiprocessing as mp
from pathlib import Path

DRIVEWAM_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
)
if DRIVEWAM_ROOT not in sys.path:
    sys.path.insert(0, DRIVEWAM_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


from navsim.agents.negwam.negwam_config import NegwamConfig
from navsim.agents.negwam.negwam_dataset import NavSimEpisodeDataset
from navsim.agents.negwam.evaluator.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id, init_logger, logger

from navsim.agents.negwam.modules.navsim_model import WanTransformer3DNavSimModel
from navsim.agents.negwam.modules.utils import load_text_encoder, load_tokenizer, load_vae
from navsim.evaluate.pdm_score import pdm_score as _pdm_score_fn
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.common.dataclasses import Trajectory, PDMResults
from navsim.common.dataloader import MetricCacheLoader
_PDM_AVAILABLE = True



class NavSimEvaluator:
    def __init__(self, config, checkpoint_path, attn_mode="torch", device=None):
        if attn_mode == "flex":
            raise ValueError(
                "Rollout evaluation uses the single-stream inference path. "
                "Use attn_mode='torch' or 'flashattn'."
            )

        self.config = config
        self.device = torch.device(device) if device is not None else torch.device("cuda:0")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.cache_name = "navsim_eval_rollout"

        logger.info(f"Loading transformer from {checkpoint_path} with attn_mode={attn_mode} ...")
        # transformer_path = os.path.join(checkpoint_path, "transformer")
        transformer_path = checkpoint_path
        self.transformer = WanTransformer3DNavSimModel.from_pretrained(
            transformer_path,
            torch_dtype=self.dtype,
            attn_mode=attn_mode,
        ).to(self.device).eval()
        self.transformer.requires_grad_(False)

        logger.info("Loading VAE ...")
        self.vae = load_vae(
            os.path.join(config.wan22_pretrained_model_name_or_path, "vae"),
            torch_dtype=self.dtype,
            torch_device=self.device,
        ).eval()
        self.vae.requires_grad_(False)

        self.video_scheduler = FlowMatchScheduler(
            shift=config.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

        cache_path = Path(self.config.dataset_path).parent.parent / "text_embed.pt"
        self._prompt_cache = {
            k: v.to(dtype=self.dtype, device=self.device)
            for k, v in torch.load(cache_path, map_location=f"cuda:{config.local_rank}", weights_only=True).items()}
        # self._prompt_cache = {}

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_images(self, images):
        """images: (B, T, C, H, W) uint8 -> normalized latents (B, C, T, H', W')."""
        batch_size, num_frames = images.shape[:2]
        imgs = F.interpolate(
            images.float().flatten(0, 1),
            size=(self.config.height, self.config.width),
            mode="bilinear",
            align_corners=False,
        )
        imgs = rearrange(imgs, "(b t) c h w -> b t c h w", b=batch_size, t=num_frames)
        videos = rearrange(imgs / 255.0 * 2.0 - 1.0, "b t c h w -> b c t h w").to(self.dtype)

        mu = self.vae.encode(videos).latent_dist.mean
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=mu.device, dtype=mu.dtype).view(
            1, -1, 1, 1, 1
        )
        latents_std = torch.tensor(self.vae.config.latents_std, device=mu.device, dtype=mu.dtype).view(
            1, -1, 1, 1, 1
        )
        return ((mu.float() - latents_mean) / latents_std).to(self.dtype)

    @torch.no_grad()
    def _encode_prompt(self, prompts):
        embeds = []
        for prompt in prompts:
            embeds.append(self._prompt_cache[prompt])
        return torch.stack(embeds, dim=0)

    def _rng_devices(self):
        if self.device.type == "cuda" and self.device.index is not None:
            return [self.device.index]
        return []

    def _make_video_input(self, latents, timesteps, prompt_emb, ego_cond, frame_index):
        grid_id = get_mesh_id(
            latents.shape[-3] // self.patch_size[0],
            latents.shape[-2] // self.patch_size[1],
            latents.shape[-1] // self.patch_size[2],
            t=0,
            f_shift=frame_index,
            action=False,
        ).to(self.device)[None]
        return {  # latents, timesteps, grid_id, text_emb, ego_cond
            "noisy_latents": latents,
            "timesteps": timesteps,
            "grid_id": grid_id,
            "text_emb": prompt_emb,
            "ego_cond": ego_cond,
        }

    def _make_action_input(self, actions, timesteps, prompt_emb, ego_cond, chunk_index):
        grid_id = get_mesh_id(
            actions.shape[-3],
            actions.shape[-2],
            actions.shape[-1],
            t=1,
            f_shift=chunk_index,
            action=True,
        ).to(self.device)[None]
        return {
            "noisy_latents": actions,
            "timesteps": timesteps,
            "grid_id": grid_id,
            "text_emb": prompt_emb,
            "ego_cond": ego_cond,
        }

    def _prepare_timesteps(self, scheduler, num_steps):
        scheduler.set_timesteps(num_steps)
        return F.pad(scheduler.timesteps, (0, 1), mode="constant", value=0)

    def _prepare_video_timesteps(self, num_steps, denoise_fraction=1.0):
       
        stop_sigma_raw = max(0.0, 1.0 - denoise_fraction)
        raw_sigmas = torch.linspace(1.0, stop_sigma_raw, num_steps + 1)

        shift = self.video_scheduler.shift
        shifted_sigmas = shift * raw_sigmas / (1 + (shift - 1) * raw_sigmas)

        self.video_scheduler.sigmas = shifted_sigmas
        self.video_scheduler.timesteps = (
            shifted_sigmas * self.video_scheduler.num_train_timesteps
        )

        return self.video_scheduler.timesteps

    @torch.no_grad()
    def _init_rollout_cache(self, first_frame_latent, action_cond):
        latent_token_per_chunk = (  # (t * h * w // patch_size)
            first_frame_latent.shape[-3] * first_frame_latent.shape[-2] * first_frame_latent.shape[-1]
        ) // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        action_token_per_chunk = action_cond.shape[-3] * action_cond.shape[-2] * action_cond.shape[-1]  # patch_size = 1 (t * n)
        self.transformer.create_empty_cache(
            self.cache_name,  # cache_name: "navsim_eval_rollout"
            self.config.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            device=self.device,
            dtype=self.dtype,
            batch_size=first_frame_latent.shape[0],
        )

    @torch.no_grad()
    def _prime_rollout_cache(self, first_frame_latent, action_cond, prompt_emb, ego_cond):
        self._init_rollout_cache(first_frame_latent, action_cond)  # first_frame_latent: (B, C, T, H', W')

        zero_t = torch.zeros((1, 1), device=self.device, dtype=torch.float32)

        video_input = self._make_video_input(
            first_frame_latent,
            zero_t,
            prompt_emb,
            ego_cond,
            frame_index=0,
        )
        self.transformer(
            video_input,
            update_cache=2,
            cache_name=self.cache_name,
            action_mode=False,
        )

        action_input = self._make_action_input(
            action_cond,
            zero_t,
            prompt_emb,
            ego_cond,
            chunk_index=0,
        )
        self.transformer(
            action_input,
            update_cache=2,
            cache_name=self.cache_name,
            action_mode=True,
        )

    @torch.no_grad()
    def _rollout_video_frame(self, frame_index, latent_shape, prompt_emb, ego_cond, num_steps,
                             denoise_fraction=1.0):
        latents = torch.randn(
            latent_shape,
            device=self.device,
            dtype=self.dtype,
        )
        timesteps = self._prepare_video_timesteps(num_steps, denoise_fraction)

        for step_id, t in enumerate(timesteps):
            last_step = step_id == len(timesteps) - 1
            timestep_tensor = torch.full(
                (1, latents.shape[2]),
                float(t.item()),
                device=self.device,
                dtype=torch.float32,
            )
            input_dict = self._make_video_input(
                latents,
                timestep_tensor,
                prompt_emb,
                ego_cond,
                frame_index=frame_index,
            )
            latent_pred_seq = self.transformer(
                input_dict,
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=False,
            )
            if last_step:
                continue

            latent_pred = data_seq_to_patch(
                self.patch_size,
                latent_pred_seq,
                latents.shape[-3],
                latents.shape[-2],
                latents.shape[-1],
                batch_size=latents.shape[0],
            )
            latents = self.video_scheduler.step(latent_pred, t, latents)

        return latents

    @torch.no_grad()
    def _rollout_action_chunk(self, chunk_index, action_shape, action_mask, prompt_emb, ego_cond, num_steps):
        actions = torch.randn(
            action_shape,
            device=self.device,
            dtype=self.dtype,
        )
        actions = torch.where(action_mask, actions, torch.zeros_like(actions))
        timesteps = self._prepare_timesteps(self.action_scheduler, num_steps)

        for step_id, t in enumerate(timesteps):
            last_step = step_id == len(timesteps) - 1
            actions = torch.where(action_mask, actions, torch.zeros_like(actions))
            timestep_tensor = torch.full(
                (1, actions.shape[2]),
                float(t.item()),
                device=self.device,
                dtype=torch.float32,
            )
            input_dict = self._make_action_input(
                actions,
                timestep_tensor,
                prompt_emb,
                ego_cond,
                chunk_index=chunk_index,
            )
            action_pred_seq = self.transformer(
                input_dict,
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=True,
            )
            if last_step:
                continue

            action_pred = rearrange(
                action_pred_seq,
                "b (f n) c -> b c f n 1",
                f=actions.shape[2],
            )
            actions = self.action_scheduler.step(action_pred, t, actions)

        return torch.where(action_mask, actions, torch.zeros_like(actions))

    def pred_traj_to_poses(self, pred_traj):
        """
        pred_traj: np.ndarray (B, 3, 8) = [x, y, heading] cumulative in ego-local frame.
        Returns: np.ndarray (8, 3) for PDM scoring.
        """
        return pred_traj[0].T.astype(np.float32)  # (8, 3)

    def actions_to_trajectory(self, actions):
        """
        Quantile-normalized action tensor -> absolute trajectory.
        actions: (B, action_dim, chunk_count=2, chunk_steps=8, 1)
        Returns: np.ndarray (B, 3, 8), channels = [x, y, heading], ego-local frame.
        """
        action_channels = list(self.config.used_action_channel_ids)  # [0, 1, 2]
        deltas = actions[:, action_channels, 1, :, 0].float()  # (B, 3, 8)
        q01 = torch.tensor(
            list(self.config.norm_stat["q01"]),
            device=deltas.device,
            dtype=deltas.dtype,
        )
        q99 = torch.tensor(
            list(self.config.norm_stat["q99"]),
            device=deltas.device,
            dtype=deltas.dtype,
        )
        
        deltas = (deltas + 1.0) / 2.0 * (q99 - q01)[:, None] + q01[:, None]
        return torch.cumsum(deltas, dim=-1).cpu().numpy()  # (B, 3, 8)

    # ------------------------------------------------------------------
    # Rollout inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, batch, num_video_steps=20, num_action_steps=20, seed=0,
                video_denoise_fraction=1.0):
        """
        Autoregressive rollout for one sample.

        Video:
            frame 0 is cached as clean context, then future frames are generated one by one.
        Action:
            chunk 0 is cached as the clean zero condition, then chunk 1 is sampled.

        Returns:
            pred_actions: (1, action_dim, 2, chunk_steps, 1)
            pred_latents: (1, C, T, H', W')
        """
        images = batch["images"][:1].to(self.device)
        if isinstance(batch["prompt"], list):
            prompt = batch["prompt"][:1]
        else:
            prompt = [batch["prompt"][0]]
        ego_cond = {
            "velocity_acc": batch["ego_velocity_acc"][:1].to(self.device),
            "history_poses": batch["history_poses"][:1].to(self.device),
            "command_ids": batch["command_id"][:1].to(self.device),
        }

        first_frame_latent = self._encode_images(images[:, :1]).to(self.dtype)
        prompt_emb = self._encode_prompt(prompt)

        action_cond = torch.zeros_like(batch["actions"][:1, :, 0:1], device=self.device, dtype=self.dtype)
        future_action_mask = batch["actions_mask"][:1, :, 1:2].bool().to(self.device)

        predicted_latents = [first_frame_latent]
        rng_devices = self._rng_devices()

        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed)
            self._prime_rollout_cache(first_frame_latent, action_cond, prompt_emb, ego_cond)  # rollout KV cache

            try:
                future_latent_shape = (
                    first_frame_latent.shape[0],
                    first_frame_latent.shape[1],
                    1,
                    first_frame_latent.shape[-2],
                    first_frame_latent.shape[-1],
                )

                num_future_latents = (images.shape[1] - 1) // 4  # (5 - 1) / 4 = 1
                for frame_index in range(1, 1 + num_future_latents):
                    frame_latent = self._rollout_video_frame(
                        frame_index=frame_index,
                        latent_shape=future_latent_shape,
                        prompt_emb=prompt_emb,
                        ego_cond=ego_cond,
                        num_steps=num_video_steps,
                        denoise_fraction=video_denoise_fraction,
                    )
                    predicted_latents.append(frame_latent)

                future_action = self._rollout_action_chunk(  # TODO: (zzh) video latent, kv cache, cross attention, action
                    chunk_index=1,
                    action_shape=(
                        action_cond.shape[0],
                        action_cond.shape[1],
                        1,
                        action_cond.shape[-2],
                        action_cond.shape[-1],
                    ),
                    action_mask=future_action_mask,
                    prompt_emb=prompt_emb,
                    ego_cond=ego_cond,
                    num_steps=num_action_steps,
                )
            finally:
                self.transformer.clear_cache(self.cache_name)

        pred_latents = torch.cat(predicted_latents, dim=2)
        pred_actions = torch.cat([action_cond, future_action], dim=2)
        return pred_actions, pred_latents

# =============== other branch ================
def _worker_fn(rank, args, result_dir):
    """Per-GPU evaluation worker. Called by mp.spawn with rank in [0, num_gpus)."""
    device = f"cuda:{rank}"

    config = NegwamConfig()
    config.dataset_path = args.dataset_path
    config.rank = rank
    config.local_rank = rank
    config.world_size = args.num_gpus
    evaluator = NavSimEvaluator(
        config,
        checkpoint_path=args.checkpoint,
        attn_mode="flashattn",
        device=device,
    )

    dataset = NavSimEpisodeDataset(config=config)
    total = (
        min(len(dataset), args.num_eval_samples)
        if args.num_eval_samples > 0
        else len(dataset)
    )

    # Shard: rank handles indices rank, rank+num_gpus, rank+2*num_gpus, ...
    indices = list(range(rank, total, args.num_gpus))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # PDM scoring setup (CPU, independent per process)
    pdm_enabled = os.path.isdir(args.metric_cache_path)  # args.metric_cache_path is not None
    if pdm_enabled and _PDM_AVAILABLE:
        proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        simulator = PDMSimulator(proposal_sampling=proposal_sampling)
        scorer = PDMScorer(proposal_sampling=proposal_sampling)
        metric_cache_loader = MetricCacheLoader(Path(args.metric_cache_path))
        traj_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    else:
        pdm_enabled = False

    rank_results = []
    for local_idx, batch in enumerate(tqdm(loader, desc=f"GPU{rank}", position=rank, leave=True)):
        global_idx = indices[local_idx]
        pred_actions, pred_latents = evaluator.predict(
            batch,
            num_video_steps=args.num_video_steps,
            num_action_steps=args.num_action_steps,
            seed=args.seed,
            video_denoise_fraction=args.video_denoise_fraction,
        )
        pred_traj = evaluator.actions_to_trajectory(pred_actions)

        pdm_result = None
        if pdm_enabled:
            token = batch["scene_token"][0]
            if token in metric_cache_loader.metric_cache_paths:
                with lzma.open(metric_cache_loader.metric_cache_paths[token], "rb") as f:
                    metric_cache = pickle.load(f)
                poses = evaluator.pred_traj_to_poses(pred_traj)
                trajectory = Trajectory(poses=poses, trajectory_sampling=traj_sampling)
                try:
                    pdm_result = _pdm_score_fn(
                        metric_cache=metric_cache,
                        model_trajectory=trajectory,
                        future_sampling=proposal_sampling,
                        simulator=simulator,
                        scorer=scorer,
                    )
                except Exception as e:
                    print(f"[GPU{rank}] PDM failed for token {token}: {e}")

        row = {
            "global_idx": global_idx,
            "token": batch["scene_token"][0],
            "pdm_valid": pdm_result is not None,
        }
        if pdm_result is not None:
            row.update(dataclasses.asdict(pdm_result))
        rank_results.append(row)
    
    # Path(result_dir).mkdir(parents=True, exist_ok=True)
    # torch.save(rank_results, Path(result_dir) / f"rank_{rank}.pt")


def evaluate(args):
    # Validate GPU count
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        raise RuntimeError(f"Requested {args.num_gpus} GPUs but only {available_gpus} available.")

    # result_dir = tempfile.mkdtemp(prefix="navsim_eval_")
    result_dir = args.output_path
    # Path(result_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Spawning {args.num_gpus} GPU workers. "
        f"video_steps={args.num_video_steps}, action_steps={args.num_action_steps}, "
        f"video_denoise_fraction={args.video_denoise_fraction}"
    )

    if args.num_gpus > 1:
        mp.spawn(_worker_fn, args=(args, result_dir), nprocs=args.num_gpus, join=True)
    else:
        _worker_fn(0, args, result_dir)

    # Aggregate results from all ranks
    all_results = []
    for rank in range(args.num_gpus):
        rank_results = torch.load(Path(result_dir) / f"rank_{rank}.pt", weights_only=False)
        all_results.extend(rank_results)
    all_results.sort(key=lambda r: r["global_idx"])

    print(f"\n=== NavSim Parallel Evaluation ({len(all_results)} samples, {args.num_gpus} GPUs) ===")
    print(f"  video_steps={args.num_video_steps}  action_steps={args.num_action_steps}  "
          f"video_denoise_fraction={args.video_denoise_fraction}")

    pdm_results_list = [r for r in all_results if r.get("pdm_valid")]
    print(f"failure pdm token number: {len(all_results) - len(pdm_results_list)}")
    if pdm_results_list and _PDM_AVAILABLE:
        fields = dataclasses.fields(PDMResults)
        print(f"\n=== PDM Score ({len(pdm_results_list)} samples) ===")
        for field in fields:
            vals = [r[field.name] for r in pdm_results_list]
            print(f"  {field.name}: {np.mean(vals):.4f}")


def main():
    parser = argparse.ArgumentParser(description="NavSim rollout evaluation")
    parser.add_argument("--config-name", default="navsim_cfg")
    parser.add_argument("--checkpoint", type=str, 
                        default=None)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data/navsim/test",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=8,
        help="Number of GPUs to use for parallel inference.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-eval-samples", type=int, default=-1,
        help="Number of samples to evaluate. Use -1 for all.",
    )
    parser.add_argument("--num-video-steps", type=int, default=3,
                        help="Video denoising steps.")
    parser.add_argument("--num-action-steps", type=int, default=10,
                        help="Action denoising steps.")
    parser.add_argument(
        "--video-denoise-fraction", type=float, default=0.6,
        help="Partial video denoising (0.0–1.0). "
             "1.0 = fully denoise to sigma=0 (default action-style).",
    )
    parser.add_argument(
        "--metric-cache-path",
        type=str,
        default=None,
        help="Path to metric cache directory for PDM score computation.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="xxx/exp",
        help="Path to save the evaluation results.",
    )
    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    init_logger()
    main()
