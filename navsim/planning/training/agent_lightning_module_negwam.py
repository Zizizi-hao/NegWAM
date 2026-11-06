import time
from typing import Dict, Tuple

import pytorch_lightning as pl
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self._train_start_time = None

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features = batch
        input_dict, prediction = self.agent.forward(features)
        loss_dict = self.agent.compute_loss(input_dict, prediction)

        # ============== loss ================
        total_loss = loss_dict["total_loss"]
        latent_loss = loss_dict["latent_loss"]
        action_loss = loss_dict["action_loss"]

        is_train = logging_prefix == "train"
        batch_size = features["images"].shape[0]

        self.log(
            f"{logging_prefix}/total_loss",
            total_loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=True,
            sync_dist=not is_train,
            batch_size=batch_size,
        )
        self.log(
            f"{logging_prefix}/latent_loss",
            latent_loss,
            on_step=is_train,
            on_epoch=not is_train,
            batch_size=batch_size,
        )
        self.log(
            f"{logging_prefix}/action_loss",
            action_loss,
            on_step=is_train,
            on_epoch=not is_train,
            batch_size=batch_size,
        )

        opt = self.optimizers() if is_train else None
        if opt:
            lr = opt.param_groups[0]["lr"]
            self.log(f"{logging_prefix}/lr", lr, on_step=True, on_epoch=False)
        
        if self.global_rank == 0 and self.global_step % 100 == 0:
            total_steps = self.trainer.max_steps
            current_step = self.global_step
            elapsed_seconds = time.time() - self._train_start_time
            seconds_per_step = elapsed_seconds / max(current_step, 1)
            remaining_steps = max(total_steps - current_step, 0)
            eta_seconds = seconds_per_step * remaining_steps

            if self.training:
                print(
                f"[Step {current_step}/{total_steps}] "
                f"ETA: {self._format_eta(eta_seconds)} | "
                f"{seconds_per_step:.2f}s/step | "
                f"latent_loss={latent_loss:.4f}, "
                f"action_loss={action_loss:.4f}, "
                f"total_loss={total_loss:.4f}"
                )

        return total_loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        """
        Configure the gradient clipping
        """
        if not gradient_clip_val:
            return 
        self.agent._negwam_model.configure_gradient_clipping(gradient_clip_val)


    def on_train_start(self):
        """
        Called when the training starts
        """
        self._train_start_time = time.time()
    
    def _format_eta(self, seconds: float) -> str:
        """
        Format the ETA in a human-readable format
        :param seconds: ETA in seconds
        :return: ETA in a human-readable format
        """
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        return f"{minutes}m{secs:02d}s"