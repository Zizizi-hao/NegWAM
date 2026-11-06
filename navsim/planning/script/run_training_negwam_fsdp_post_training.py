import os
import gc
import json
import logging
import pickle
import traceback
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
import swanlab as wandb

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, DistributedSampler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.negwam.negwam_dataset import NavSimEpisodeDataset
from navsim.agents.negwam.distributed.util import configure_model, init_distributed, dist_mean, dist_max
from navsim.agents.negwam.distributed.fsdp import apply_ac, shard_model

from torch.distributed.checkpoint.state_dict import(
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from safetensors.torch import save_file


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training_negwam"

def build_datasets(cfg: DictConfig, agent: AbstractAgent):
    """
    traindata /valdata 
    """
    dataset = NavSimEpisodeDataset(agent._config)

    dataloader_params = dict(cfg.dataloader.params)
    if dataloader_params.get("num_workers", 0) > 0:
        dataloader_params.setdefault("persistent_workers", True)

    train_sampler = (
        DistributedSampler(
            dataset,
            num_replicas=cfg.world_size,
            rank=cfg.rank,
            shuffle=True,
            seed=cfg.seed,
        )
        if cfg.world_size > 1 else None
    )

    train_loader_params = {k: v for k, v in dataloader_params.items() if k != "shuffle"}

    logger.info("Building Dataloaders")
    train_dataloader = DataLoader(
        dataset, 
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        **train_loader_params
        )
    logger.info("Num training samples: %d", len(dataset))

    return train_dataloader, train_sampler


class Trainer:
    def __init__(self, config: DictConfig, agent: AbstractAgent):
        if config.enable_wandb and config.rank == 0:
            wandb.login(api_key=os.environ["WANDB_API_KEY"])
            self.wandb = wandb
            self.wandb.init(
                project="wam-rl",
                workspace="Darcy-cheng",
                mode="local",  
                name="navsim_train",
            )
            logger.info("WandB logging enabled")

        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = agent._config.param_dtype
        self.step = 0
        self.max_steps = config.trainer.params.max_steps
        self.config = config
        self.agent = agent
        self.grad_accumulation_steps = config.trainer.params.accumulate_grad_batches
        self.model = agent._negwam_model.transformer
        self.gc_interval = config.trainer.params.gc_interval
        self.save_interval = config.trainer.params.save_interval
        self.save_root = Path(agent._config.save_root)


        self.train_loader, self.train_sampler = build_datasets(self.config, self.agent)
        self.train_loader_iter = None

        # FSDP
        logger.info("Setting up activation checkpointing...")
        apply_ac(self.model)
        
        logger.info("Setting up FSDP...")
        self.model = configure_model(
            model=self.model,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )

        opt_cfg = agent.get_optimizers()
        self.optimizer = opt_cfg["optimizer"]
        self.scheduler = opt_cfg["lr_scheduler"]["scheduler"]
        

    def train(self):
        """
        Train the model using FSDP
        :param cfg: omegaconf dictionary
        :param agent: agent interface in NAVSIM
        """
        logger.info(f"Starting NavSim training for {self.max_steps} steps...")
        progress_bar = tqdm(
            total=self.max_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
            mininterval=1,  
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_repulsion_losses = []
        accumulated_compat_losses = []
        step_in_accumulation = 0

        while self.step < self.max_steps:
            train_batch = self.get_next_batch(self.train_loader, is_train=True)
            losses = self.train_step(train_batch, step_in_accumulation)

            accumulated_latent_losses.append(losses["latent_loss"])
            accumulated_action_losses.append(losses["action_loss"])
            accumulated_repulsion_losses.append(losses["repulsion_loss"])
            accumulated_compat_losses.append(losses["compat_loss"])
            step_in_accumulation += 1

            if losses["should_log"]:
                self.log_losses(progress_bar, losses,
                    accumulated_latent_losses,
                    accumulated_action_losses,
                    accumulated_repulsion_losses,
                    accumulated_compat_losses,
                )

                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_repulsion_losses = []
                accumulated_compat_losses = []
                step_in_accumulation = 0

                # torch.cuda.synchronize()
                if self.step % self.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                

                self.step += 1

                if self.step % self.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Saving checkpoint at step {self.step}")
                    
                    self.save_checkpoint(resume_training=self.config.resume_training)

        dist.destroy_process_group()
        progress_bar.close()
        logger.info("NavSim training completed!")


    def train_step(self, batch, batch_idx):

        batch = self.move_to_device(batch)
        
        should_sync = (batch_idx + 1) % self.grad_accumulation_steps == 0
        if not should_sync:
            self.model.set_requires_gradient_sync(False)
        else:
            self.model.set_requires_gradient_sync(True)

        input_dict, prediction = self.agent.forward(batch)
        loss_dict = self.agent.compute_loss(input_dict, prediction)

        total_loss = loss_dict["total_loss"]

        latent_loss_log = loss_dict["latent_loss"].detach().float().cpu().item()
        action_loss_log = loss_dict["action_loss"].detach().float().cpu().item()
        repulsion_loss_log = loss_dict["repulsion_loss"].detach().float().cpu().item()
        compat_loss_log = loss_dict["compat_loss"].detach().float().cpu().item()
        total_loss.backward()

        losses = {"latent_loss": latent_loss_log,
                "action_loss": action_loss_log,
                "repulsion_loss": repulsion_loss_log,
                "compat_loss": compat_loss_log}

        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.trainer.params.gradient_clip_val)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            losses["total_norm"] = total_norm
            losses["should_log"] = True
        else:
            losses["should_log"] = False

        return losses


    def get_next_batch(self, dataloader: DataLoader, is_train: bool=True):
        iter_attr = "train_loader_iter" if is_train else "val_loader_iter"
        loader_iter = getattr(self, iter_attr)

        if loader_iter is None:
            loader_iter = iter(dataloader)
        
        try:
            batch = next(loader_iter)
        
        except StopIteration:
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(dataloader.sampler.epoch + 1)
            loader_iter = iter(dataloader)
            batch = next(loader_iter)

        setattr(self, iter_attr, loader_iter)
        return batch

    def move_to_device(self, item):
        if torch.is_tensor(item):
            return item.to(self.device)
        if isinstance(item, dict):
            return {k: self.move_to_device(v) for k, v in item.items()}
        return item

    def log_losses(self, progress_bar, losses,
                    accumulated_latent_losses,
                    accumulated_action_losses,
                    accumulated_repulsion_losses,
                    accumulated_compat_losses, eval=False):

        if eval:
            logger.info(f"Evaluating model at step {self.step}")
            logger.info(f"Validation loss: {losses['total_loss'].item():.4f}")
            logger.info(f"Validation latent loss: {losses['latent_loss'].item():.4f}")
            logger.info(f"Validation action loss: {losses['action_loss'].item():.4f}")
            logger.info(f"Validation repulsion loss: {losses['repulsion_loss'].item():.4f}")
            logger.info(f"Validation compat loss: {losses['compat_loss'].item():.4f}")
            return

        lr = self.scheduler.get_last_lr()[0]

        latent_loss_show = dist_mean(torch.tensor(sum(accumulated_latent_losses), device=self.device)).item()
        action_loss_show = dist_mean(torch.tensor(sum(accumulated_action_losses), device=self.device)).item()
        repulsion_loss_show = dist_mean(torch.tensor(sum(accumulated_repulsion_losses), device=self.device)).item()
        compat_loss_show = dist_mean(torch.tensor(sum(accumulated_compat_losses), device=self.device)).item()
        max_latent_loss_show = dist_max(torch.tensor(max(accumulated_latent_losses), device=self.device)).item()
        max_action_loss_show = dist_max(torch.tensor(max(accumulated_action_losses), device=self.device)).item()
        max_repulsion_loss_show = dist_max(torch.tensor(max(accumulated_repulsion_losses), device=self.device)).item()
        if self.config.rank == 0:
            total_norm = losses["total_norm"]
            progress_bar.n += 1
            progress_bar.set_postfix(
                {
                    "latent_loss": f"{latent_loss_show:.4f}",
                    "action_loss": f"{action_loss_show:.4f}",
                    "repulsion_loss": f"{repulsion_loss_show:.4f}",
                    "compat_loss": f"{compat_loss_show:.4f}",
                    "step": self.step,
                    "grad_norm": f"{total_norm.item():.2f}",
                    "lr": f"{lr:.2e}",
                }
            )
            if self.config.enable_wandb:
                log_dict = {
                    "loss_metrics/global_avg_video_loss": latent_loss_show,
                    "loss_metrics/global_avg_action_loss": action_loss_show,
                    "loss_metrics/global_avg_repulsion_loss": repulsion_loss_show,
                    "loss_metrics/global_avg_compat_loss": compat_loss_show,
                    "loss_metrics/global_max_video_loss": max_latent_loss_show,
                    "loss_metrics/global_max_action_loss": max_action_loss_show,
                    "loss_metrics/global_max_repulsion_loss": max_repulsion_loss_show,
                    "grad_norm": total_norm.item(),
                    "lr": lr,
                }
                self.wandb.log(log_dict, step=self.step)

    def save_checkpoint(self, resume_training=False):
        """
        Save the model checkpoint
        :param resume_training
        """
        if resume_training:
            self.save_checkpoint_resume()
        else:
            self.save_checkpoint_raw()


    def save_checkpoint_resume(self):
        """
        Save the model checkpoint for resume training
        """
        try:
            state_dict_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            model_state = get_model_state_dict(self.model, options=state_dict_options)
            model_state_bf16 = {k: v.to(torch.bfloat16) for k, v in model_state.items()}

            optimizer_state = get_optimizer_state_dict(self.model, self.optimizer, options=state_dict_options)

            training_state = {
            "step": self.step,
            "world_size": self.config.world_size,
            "model": model_state_bf16,
            "optimizer": optimizer_state,
            "scheduler": self.scheduler.state_dict(),
            "sampler_epoch": self.train_sampler.epoch if self.train_sampler else 0,
        }

            if self.config.rank == 0:
                checkpoint_dir = self.save_root / "checkpoints" / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self.save_state_dict(model_state_bf16, checkpoint_dir)

                last_dir = self.save_root / "checkpoints" / "last"
                last_dir.mkdir(parents=True, exist_ok=True)
                torch.save(training_state, last_dir / "training_state.pt")
                logger.info(f"Resume checkpoint saved at step {self.step}: {last_dir}")

            if dist.is_initialized():
                dist.barrier()
        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save resume checkpoint: {e}")
                logger.error(traceback.format_exc())
            if dist.is_initialized():
                dist.barrier()
            raise

    def save_checkpoint_raw(self):
        """
        Save the model checkpoint for resume training
        """
        try:
            state_dict = get_model_state_dict(
                self.model, 
                options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            if self.config.rank == 0:
                checkpoint_dir = self.save_root / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self.save_state_dict(state_dict_bf16, checkpoint_dir)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            if dist.is_initialized():
                dist.barrier()
            
        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                logger.error(traceback.format_exc())
            if dist.is_initialized():
                dist.barrier()

    def save_state_dict(self, state_dict, checkpoint_dir):
        transformer_dir = checkpoint_dir / "transformer"
        transformer_dir.mkdir(parents=True, exist_ok=True)
        model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
        save_file(state_dict, model_file)

        config_file = transformer_dir / "config.json"
        config_dict = dict(self.model.config)
        config_dict.pop("_name_or_path", None)
        config_dict["_class_name"] = "WanTransformer3DNavSimModel"
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)
    

    def load_checkpoint_resume(self, resume_path: str):
        state_dict_options = StateDictOptions(full_state_dict=True, cpu_offload=True)

        checkpoint = torch.load(resume_path, map_location="cpu")

        set_model_state_dict(self.model, checkpoint["model"], options=state_dict_options)
        
        opt_cfg = self.agent.get_optimizers()
        self.optimizer = opt_cfg["optimizer"]
        self.scheduler = opt_cfg["lr_scheduler"]["scheduler"]
        
        
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            checkpoint["optimizer"],
            options=state_dict_options,
        )
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step = checkpoint["step"]

        if self.config.rank == 0:
            logger.info(f"Resumed from {resume_path}, step={self.step}")

        if self.train_sampler is not None and "sampler_epoch" in checkpoint:
            self.train_sampler.set_epoch(checkpoint["sampler_epoch"])
        self.train_loader_iter = None



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    """
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    machine_num = int(os.environ.get("DISTRIBUTED_NODE_COUNT", 1))
    master_addr = os.environ.get("DISTRIBUTED_MASTER_HOSTS", "127.0.0.1")
    tcp_port = os.environ.get("LUBAN_AVAILBLE_PORT_0", None)
    single_machine = machine_num == 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    OmegaConf.set_struct(cfg, False)
    cfg.rank = rank
    cfg.local_rank = local_rank
    cfg.world_size = world_size

    if single_machine:
        init_distributed(world_size, local_rank, rank)
        logger.info(f"Initialized distributed training with local rank {local_rank}")
    else:
        dist_url = f"tcp://{master_addr}:{tcp_port}"

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            rank=rank,
            world_size=world_size,
            device_id=local_rank,
        )
        logger.info(
            f"Initialized distributed training with local rank {local_rank} global rank {rank}, dist_url {dist_url}"
        )

    
    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    trainer = Trainer(cfg, agent)
    trainer.train()



if __name__ == "__main__":
    main()
