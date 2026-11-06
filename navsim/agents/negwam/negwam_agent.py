from typing import Any, Dict, List, Optional, Union
import pickle, json
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import default_collate
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar

# navsim_agent
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.negwam.negwam_config import NegwamConfig
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.negwam.evaluator.evaluator import NavSimEvaluator
from navsim.agents.negwam.negwam_dataset import NavSimEpisodeDataset
from navsim.agents.negwam.negwam_model import NegwamModel
from navsim.agents.negwam.negwam_loss import NegwamLoss
from navsim.agents.negwam.evaluator.utils import warmup_constant_lambda

import logging
logger = logging.getLogger(__name__)

class NegwamAgent(AbstractAgent):
    """Agent interface for NegWAM agent."""

    def __init__(
        self,
        config: NegwamConfig,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes NegWAM agent.
        :param config: global config of NegWAM agent
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        :param trajectory_sampling: trajectory sampling specification
        """
        super().__init__(trajectory_sampling)

        self._config = config
        self.requires_scene = True 
        self._checkpoint_path = checkpoint_path
        # self._negwam_model = NegwamModel(self._trajectory_sampling, config)

        self._evaluator = None
        self._token2idx = None

        self._negwam_model = NegwamModel(config)
        self._loss = NegwamLoss(config)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        if self._checkpoint_path is not None:  
            self._evaluator = NavSimEvaluator( 
                self._config, checkpoint_path=self._checkpoint_path,
                attn_mode="torch", device=device,
            )
        else:
            logger.warning("No checkpoint path provided, initializing evaluator with random weights")
    
        self._dataset = NavSimEpisodeDataset(self._config)
        self._token2idx = self._dataset.token2idx

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        token = scene.scene_metadata.initial_token          
        idx = self._token2idx[token]                        
        batch = default_collate([self._dataset[idx]])        
        with torch.no_grad():
            pred_actions, _ = self._evaluator.predict(batch, num_video_steps=self._config.num_video_steps, num_action_steps=self._config.num_action_steps)
        pred_traj = self._evaluator.actions_to_trajectory(pred_actions)   # (1,3,8)
        poses = self._evaluator.pred_traj_to_poses(pred_traj)             # (8,3)
        return Trajectory(poses=poses, trajectory_sampling=self._trajectory_sampling)

    
    # ============= traning part =============
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._negwam_model(features)

    def compute_loss(
        self,
        input_dict: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        latent_loss, action_loss, repulsion_loss, compat_loss = self._loss(input_dict, predictions)

        repulsion_loss = self._config.repulsion_lambda * repulsion_loss   
        compat_loss = self._config.compat_lambda * compat_loss   

        total_loss = latent_loss + action_loss + repulsion_loss + compat_loss
        return {
            "total_loss": total_loss,
            "latent_loss": latent_loss,
            "action_loss": action_loss,
            "repulsion_loss": repulsion_loss,
            "compat_loss": compat_loss,
        }


    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Returns the optimizers
        : freezing img backbone"""
        optimizer = torch.optim.AdamW(
            [p for p in self._negwam_model.transformer.parameters() if p.requires_grad],
            lr = self._config.learning_rate,
            betas = (self._config.beta1, self._config.beta2),
            eps = 1e-8,
            weight_decay = self._config.weight_decay,
            fused=torch.cuda.is_available(),
            foreach=False,
        )

        lr_decay_milestones = sorted(getattr(self._config, "lr_decay_milestones", []))
        lr_decay_gamma = getattr(self._config, "lr_decay_gamma", 1.0)

        def lr_lambda(step):
            scale = warmup_constant_lambda(step, warmup_steps=self._config.warmup_steps)
            for m in lr_decay_milestones:
                if step >= m:
                    scale *= lr_decay_gamma
            
            return scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1) 

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return []

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return []   

    
    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Get the training callbacks to save checkpoints
        """

        checkpoint_callback = ModelCheckpoint(   
            dirpath=f"{self._config.save_root}/checkpoints",
            filename="step={step}-loss={val_epoch/total_loss:.4f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=1, 
            every_n_train_steps=self._config.save_interval,
            save_last=True,
        )

        return [checkpoint_callback, TQDMProgressBar(refresh_rate=10)]
        

