import os
import json
import logging
import pickle
from tqdm import tqdm
from pathlib import Path
from typing import Tuple

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from pytorch_lightning.strategies import FSDPStrategy


from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module_negwam import AgentLightningModule
from navsim.agents.negwam.negwam_dataset import NavSimEpisodeDataset
from navsim.agents.negwam.modules.navsim_model import WanTransformerBlock


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training_negwam"



def build_datasets(cfg: DictConfig, agent: AbstractAgent):
    """
    traindata /valdata 
    """
    dataset = NavSimEpisodeDataset(agent._config)
    log_index_path = Path(agent._config.dataset_path) / "file2log.json"

    def create_file2log_index(log_index_path: Path):
        if log_index_path.exists():
            file2log = json.loads(log_index_path.read_text())
        else:
            file2log = {}
            for f in tqdm(dataset.sample_files, desc="Caching file2log", total=len(dataset.sample_files)):
                with open(f, "rb") as fh:
                    s = pickle.load(fh)
                log_name = s.get("metadata", {}).get("log_name", "")
                file2log[f.name] = log_name
            log_index_path.write_text(json.dumps(file2log))

        return file2log
    
    file2log = create_file2log_index(log_index_path)

    name2idx = {f.name: i for i, f in enumerate(dataset.sample_files)}
    idx2log = {name2idx[name]: log for name, log in file2log.items() if name in name2idx}   # 0: log_name, 1: log_name

    # train_data / val_data
    val_idx = [idx for idx, log in idx2log.items() if log in cfg.val_logs]
    train_idx = [idx for idx, log in idx2log.items() if log not in cfg.val_logs]

    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)

    dataloader_params = dict(cfg.dataloader.params)
    if dataloader_params.get("num_workers", 0) > 0:
        dataloader_params.setdefault("persistent_workers", True)

    logger.info("Building Dataloaders")
    train_dataloader = DataLoader(train_data, **dataloader_params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **dataloader_params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    return train_dataloader, val_dataloader



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    """
    pl.seed_everything(cfg.seed, workers=True)
    # torch.set_float32_matmul_precision("high")   
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    train_dataloader, val_dataloader = build_datasets(cfg, agent)

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    ckpt_path = cfg.get("resume_ckpt_path", None) or agent._config.checkpoint_path or None

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,   
    )


if __name__ == "__main__":
    main()
