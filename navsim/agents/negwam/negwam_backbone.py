import os
from pathlib import Path
from typing import Dict
from einops import rearrange

from torch import nn
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean

from navsim.agents.negwam.negwam_config import NegwamConfig
from navsim.agents.negwam.modules.utils import WanVAEStreamingWrapper,load_vae, load_tokenizer, load_text_encoder
from navsim.agents.negwam.evaluator.utils import FlowMatchScheduler, sample_timestep_id, get_mesh_id


# from drivewam: https://github.com/chenshi3/DriveWAM
class NegwamBackbone(nn.Module):
    """Negwam backbone
    """
    def __init__(self, config: NegwamConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size

        cache_path = Path(self.config.dataset_path).parent.parent / "text_embed.pt"
        self.prompt_embed_cache = torch.load(cache_path, map_location=f"cuda:{config.local_rank}", weights_only=True)

        # pretraining model load
        self.vae = load_vae(
            os.path.join(config.wan22_pretrained_model_name_or_path, "vae"),
            torch_dtype=torch.float32,  
            torch_device=f"cuda:{config.local_rank}"
        )

        self.vae.eval()
        self.vae.requires_grad_(False)
        self.steaming_vae = WanVAEStreamingWrapper(self.vae)

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(
            shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.train_scheduler_action.set_timesteps(1000, training=True)



    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Negwam backbone
        :param features: dictionary of features
        :return: dictionary of predictions
        """
        latent_outputs = self._prepare_input_dict(features)

        return latent_outputs

    
    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """
        input:batch_dict (['images', 'prompt', 'command_id', 
        - 'ego_velocity_acc', 'history_poses', 'actions', 
        - 'actions_mask', 'scene_token', 'sample_path'])

        output:latent_dict
        - ['latent_dict', 'action_dict', 'chunk_size', 'window_size']
        """
        latents = self._encode_images_to_latents(batch_dict["images"])  # [B, T, C, H, W] ---> [B, C, T', H', W']
        latent_mask = self._build_latent_prediction_mask(latents)
        text_emb = self._encode_prompts(batch_dict["prompt"])  

        latent_dict = self._add_noise( 
            latent=latents,
            train_scheduler=self.train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=0.5,
            noise_mask=latent_mask,
        )
        action_dict = self._add_noise(  
            latent=batch_dict["actions"],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch_dict["actions_mask"],
            action_mode=True,
            noisy_cond_prob=0.0,
        )

        latent_dict["text_emb"] = text_emb
        latent_dict["latents_mask"] = latent_mask
        latent_dict["ego_cond"] = {  
            "velocity_acc": batch_dict["ego_velocity_acc"],  
            "history_poses": batch_dict["history_poses"],
            "command_ids": batch_dict["command_id"],
        }
        action_dict["text_emb"] = text_emb
        action_dict["actions_mask"] = batch_dict["actions_mask"]

        if "negative_trajs" in batch_dict:
            action_dict["negative_trajs"] = batch_dict["negative_trajs"]
            action_dict["negative_mask"] = batch_dict["negative_mask"]
            action_dict["negative_ade"] = batch_dict["negative_ade"]

        if "negative_actions" in batch_dict:
            gt_group = batch_dict["actions"].unsqueeze(1)          # (B,1,C,F,N,1)  GT-as-rider
            action_dict["rider_latents"] = torch.cat(
                [gt_group, batch_dict["negative_actions"]], dim=1
            )                                                       # (B, G=1+K, C, F, N, 1)
            action_dict["rider_valid"] = torch.cat(
                [
                    torch.ones_like(batch_dict["negative_mask"][:, :1]), 
                    batch_dict["negative_mask"],
                ], dim=1
            )  

        return {
            "latent_dict": latent_dict, 
            "action_dict": action_dict, 
            "chunk_size": self.config.frame_chunk_size,
            "window_size": self.config.attn_window, 
        }

    @torch.no_grad()
    def _encode_images_to_latents(self, images):

        """
        Encode images to latents
            :param images: images to encode
            :return: latents
        """
        batch_size, num_frames = images.shape[:2]
        images = images.float()
        images = F.interpolate(
            images.flatten(0, 1),
            size=(self.config.height, self.config.width),
            mode="bilinear",
            align_corners=False,
        )
        images = rearrange(images, "(b t) c h w -> b t c h w", b=batch_size, t=num_frames)
        videos = rearrange(images / 255.0 * 2.0 - 1.0, "b t c h w -> b c t h w").to(next(self.vae.parameters()).device)

        posterior = self.vae.encode(videos).latent_dist
        mu = posterior.mean

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=mu.device, dtype=mu.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std, device=mu.device, dtype=mu.dtype)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        return mu_norm
    
    def _build_latent_prediction_mask(self, latents):
        latent_mask = torch.ones(
            latents.shape[0],
            1,
            latents.shape[2],
            1,
            1,
            dtype=torch.bool,
            device=latents.device,
        )
        latent_mask[:, :, 0] = False
        return latent_mask
        

    @torch.no_grad()
    def _encode_prompts(self, prompts):
        """Encode a flat list of prompts → (len(prompts), text_max_length, text_dim)."""
        if self.config.enable_text_encoder_cache: 
            embeds = []
            for prompt in prompts:
                # if prompt not in self.prompt_embed_cache:
                #     self.prompt_embed_cache[prompt] = self._get_t5_prompt_embeds(prompt)[0]
                embeds.append(self.prompt_embed_cache[prompt])
            return torch.stack(embeds, dim=0).to(self.config.param_dtype)
        else:
            raise ValueError("enable_text_encoder_cache must be cached in the cache file")
        # return self._get_t5_prompt_embeds(prompts)



    @torch.no_grad()
    def _add_noise(
        self,
        latent,
        train_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
        noise_mask=None,
    ):
        """
        Add noise to latent or action
        :param latent: latent to add noise
        :param train_scheduler: train scheduler
        :param action_mask: action mask
        :param action_mode: action mode
        :param noisy_cond_prob: noisy cond prob
        :param noise_mask: noise mask

        Return:
        - dict(
            timesteps=timesteps,
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps,
            grid_id=latent_grid_id,
        )
        """
        B, C, F_, H, W = latent.shape
        clean_latent = latent

        timestep_ids = sample_timestep_id(batch_size=F_, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=latent.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)
        timesteps = timesteps[None].repeat(B, 1)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1,
            f_shift=0,
            action=action_mode,
        ).to(latent.device)
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                batch_size=F_,
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=train_scheduler.num_train_timesteps,
            )
            cond_noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=latent.device)
            latent = train_scheduler.add_noise(latent, cond_noise, cond_timesteps, t_dim=2)
            cond_timesteps = cond_timesteps[None].repeat(B, 1)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if noise_mask is not None:
            noise_frame_mask = self._dense_mask_to_frame_mask(noise_mask)
            noise_mask = noise_frame_mask[:, None, :, None, None]
            noisy_latents = torch.where(noise_mask, noisy_latents, clean_latent)
            targets = targets * noise_mask.float()
            latent = torch.where(noise_mask, latent, clean_latent)
            timesteps = torch.where(noise_frame_mask, timesteps, torch.zeros_like(timesteps))
            cond_timesteps = torch.where(noise_frame_mask, cond_timesteps, torch.zeros_like(cond_timesteps))

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()
            action_frame_mask = self._dense_mask_to_frame_mask(action_mask)
            timesteps = torch.where(action_frame_mask, timesteps, torch.zeros_like(timesteps))
            cond_timesteps = torch.where(action_frame_mask, cond_timesteps, torch.zeros_like(cond_timesteps))

        return dict(
            timesteps=timesteps,
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps,
            grid_id=latent_grid_id,
        )
    
    def _dense_mask_to_frame_mask(self, dense_mask):
        return dense_mask.bool().any(dim=1).any(dim=-1).any(dim=-1)


    # ======== helper functions ================
    def _move_to_device(self, item):
        """
        Helper function to move item to device
            :param item: item to move to device
            :return: item on device
        """
        if torch.is_tensor(item):
            return item.to(self.device)

        if isinstance(item, dict):
            return {k: self._move_to_device(v) for k, v in item.items()}

        if isinstance(item, list):
            return [self._move_to_device(v) for v in item]

        return item
    
    def normalize_latents(self, latents, latents_mean, latents_std):
        """
        Normalize latents
            :param latents: latents to normalize
            :param latents_mean: mean of latents
            :param latents_std: std of latents
            :return: normalized latents
        """
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        return ((latents.float() - latents_mean) * latents_std).to(latents)