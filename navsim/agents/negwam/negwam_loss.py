import logging
import torch
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn


from navsim.agents.negwam.negwam_config import NegwamConfig

logger = logging.getLogger(__name__)
from navsim.agents.negwam.evaluator.utils import FlowMatchScheduler, data_seq_to_patch


def negwam_loss(input_dict, pred, config):
    negwam_loss_module=NegwamLoss(config)

    latent_loss, action_loss = negwam_loss_module(input_dict, pred)

    total_loss = latent_loss + action_loss

    loss_dict = {
        "total_loss": total_loss,
        "latent_loss": latent_loss,
        "action_loss": action_loss,
    }

    return loss_dict



class NegwamLoss(nn.Module):
    def __init__(self, config: NegwamConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.gradient_accumulation_steps = config.gradient_accumulation_steps

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.train_scheduler_latent.set_timesteps(1000, training=True)

        self.train_scheduler_action = FlowMatchScheduler(
            shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.train_scheduler_action.set_timesteps(1000, training=True)

        norm_stat = config.norm_stat
        q01 = torch.tensor(norm_stat["q01"], dtype=torch.float32)
        q99 = torch.tensor(norm_stat["q99"], dtype=torch.float32)
        self.register_buffer("actions_q01", q01, persistent=False)
        self.register_buffer("actions_q99", q99, persistent=False)

    
    def forward(self, input_dict, pred):

        latent_pred, action_pred = pred[0], pred[1]
        compat_scores = pred[2] if len(pred) > 2 else None   
        action_pred = rearrange(
            action_pred,
            "b (f n) c -> b c f n 1",
            f=input_dict["action_dict"]["targets"].shape[-3],
        )
        latent_pred = data_seq_to_patch(
            self.patch_size,
            latent_pred,
            input_dict["latent_dict"]["targets"].shape[-3],
            input_dict["latent_dict"]["targets"].shape[-2],
            input_dict["latent_dict"]["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )

        latent_loss = self._flow_matching_branch_loss(
            latent_pred,
            input_dict["latent_dict"],
            mode="latent",
        )
        
        action_loss = self._flow_matching_branch_loss(
            action_pred,
            input_dict["action_dict"],
            mode="action",
        )

        repulsion_loss = self._repulsion_loss(action_pred, input_dict["action_dict"])

        compat_loss = self._compat_loss(compat_scores, input_dict["action_dict"])

        return (latent_loss / self.gradient_accumulation_steps,
                action_loss / self.gradient_accumulation_steps,
                repulsion_loss / self.gradient_accumulation_steps,
                compat_loss / self.gradient_accumulation_steps)

    
    def _flow_matching_branch_loss(
        self,
        pred,
        input_dict,
        mode: str = "latent",
    ):
        Bn, Fn = input_dict["timesteps"].shape

        scheduler = self.train_scheduler_latent if mode == "latent" else self.train_scheduler_action

        loss_weight = scheduler.training_weight(  
            input_dict["timesteps"].flatten()
        ).reshape(Bn, Fn)

        loss = F.mse_loss(
            pred.float(),
            input_dict["targets"].float().to(pred.device),
            reduction="none",
        ).to(loss_weight.device)

        loss = loss * loss_weight[:, None, :, None, None] 
        
        if mode == "latent":
            mask = input_dict["latents_mask"].float().expand_as(input_dict["targets"]).to(loss.device)
        elif mode == "action":
            mask = input_dict["actions_mask"].float().to(loss.device)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        loss = loss * mask  

        loss = loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        mask = mask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)

        loss_per_frame = loss.sum(dim=1)
        mask_per_frame = mask.sum(dim=1)

        valid = mask_per_frame > 0
        loss = (loss_per_frame[valid] / mask_per_frame[valid]).mean() 

        return loss

    
    def _timestep_to_sigma(self, scheduler, timesteps):
        """
        timesteps (B, F) -> sigma (B, F).
        """
        t = timesteps.detach().float().cpu()
        ids = torch.argmin(
            (scheduler.timesteps[None, None, :] - t[..., None]).abs(), dim=-1
        )
        return scheduler.sigmas[ids].to(timesteps.device)


    def _repulsion_loss(self, action_pred, action_dict):
        """
        action_pred: (B, C, F, N, 1)
        """
        neg = action_dict.get("negative_trajs", None)
        if neg is None:
            return torch.zeros((), device=action_pred.device)
        neg_mask = action_dict["negative_mask"].float()          # (B, K)
        if neg_mask.sum() == 0:
            return (action_pred.sum() * 0.0).abs()                      

        ch = self.config.used_action_channel_ids[:3]              # [0, 1, 2]

        sigma = self._timestep_to_sigma(
            self.train_scheduler_action, action_dict["timesteps"]
        )                                                         # (B, F)
        x0 = (
            action_dict["noisy_latents"].float()
            - sigma[:, None, :, None, None] * action_pred.float()
        )                                                         # (B, C, F, N, 1)
        x0 = x0[:, ch, 1, :, 0].permute(0, 2, 1)                  # (B, 8, 3)

        q01, q99 = self.actions_q01.to(x0.device), self.actions_q99.to(x0.device)
        delta = (x0 + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        pose = torch.cumsum(delta, dim=1)                         # (B, 8, 3)

        neg = neg.float().to(x0.device)                           # (B, K, 8, 3)
        d = (pose[:, None, :, :2] - neg[..., :2]).norm(dim=-1).mean(dim=-1)  # (B, K)

        margin = (
            self.config.repulsion_margin_alpha 
            * action_dict["negative_ade"].float().to(x0.device)
        )
        hinge = torch.relu(margin - d) * neg_mask      # (B, K)
        per_sample = hinge.sum(dim=1) / neg_mask.sum(dim=1).clamp(min=1.0)   # (B,)
        w = (1.0 - sigma[:, 1]).clamp(min=0.0)                               # (B,)
        loss = (w * per_sample).mean()

        if not torch.isfinite(loss) or loss.abs() > 1.0:
            logger.error(
                "[Repulsion] poisoned step: loss=%.3e | ade=[%.3e, %.3e] mask_sum=%.1f | "
                "d=[%.3e, %.3e] | sigma=[%.4f, %.4f] | w=[%.4f, %.4f]",
                loss.item(),
                action_dict["negative_ade"].min().item(),
                action_dict["negative_ade"].max().item(),
                neg_mask.sum().item(),
                d.min().item(), d.max().item(),
                sigma.min().item(), sigma.max().item(),
                w.min().item(), w.max().item(),
            )
            return (action_pred.sum() * 0.0).abs()
        return loss


    def _compat_loss(self, compat_scores, action_dict):

        if compat_scores is None:
            return torch.zeros((), device=action_dict["noisy_latents"].device)
        valid = action_dict["rider_valid"].bool().to(compat_scores.device)   # (B, G)
        neg_valid = valid[:, 1:]
        has_neg = neg_valid.any(dim=1)                                       # (B,)
        if has_neg.sum() == 0:
            return compat_scores.sum() * 0.0       
        logits = compat_scores / self.config.compat_temperature
        logits = logits.masked_fill(~valid, float("-inf"))
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device) 
        loss = F.cross_entropy(logits, target, reduction="none")             # (B,)
        return loss[has_neg].mean()

