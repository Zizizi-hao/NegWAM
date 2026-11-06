from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch
import os


@dataclass
class NegwamConfig:
    """NegWAM config for navsim training."""

    # ====================navsim config ====================
    latent: bool = True
    dataset_path: str = "xxx/dataset/cache/feature_cache"
    checkpoint_path: str = None

    # ==================== negative trajectory RL ======================
    negative_traj_path: Optional[str] = None   
    num_negatives: int = 16                   
    neg_min_ade_to_gt: float = 0.3             
    # repulsion_margin: float = 0.5            
    repulsion_margin_alpha: float = 0.9        
    repulsion_lambda: float = 0.1              # λ

    # =================== negative trajectory compat head ===================
    enable_compat_head: bool = False  
    compat_lambda: float = 0.05       # λ_compat
    compat_temperature: float = 0.1  
    compat_head_hidden: int = 256  

    

    # ==================== environment config ====================
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))

    # ==================== negwam config ====================
    param_dtype: torch.dtype = torch.bfloat16
    save_root: str = "xxx/exp/training_navsim"
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    wan22_pretrained_model_name_or_path: str = (
        "xxx/checkpoints/lingbot-va-base"
    )


    num_video_steps: int = 3  
    num_action_steps: int = 10  
    
    enable_wandb: bool = True
    load_worker: int = 8
    save_interval: int = 10000
    gc_interval: int = 50

    learning_rate: float = 2e-6 
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_steps: int = 100000
    lr_decay_milestones: List[int] = field(default_factory=lambda: [5000, 10000, 20000])
    lr_decay_gamma: float = 0.5

    height: int = 256
    width: int = 448
    text_max_length: int = 512

    action_dim: int = 30
    future_action_steps: int = 8
    action_chunk_count: int = 2
    action_chunk_steps: int = 8
    action_per_frame: int = 8
    future_image_indices: List[int] = field(default_factory=lambda: [1, 3, 5, 7])
    history_num_frames: int = 4
    frame_chunk_size: int = 1
    attn_window: int = 30

    snr_shift: float = 5.0
    action_snr_shift: float = 1.0

    command_prompts: List[str] = field(
        default_factory=lambda: [
            "autonomous driving scene, the intended route is to the left.",
            "autonomous driving scene, the intended route is straight.",
            "autonomous driving scene, the intended route is to the right.",
            "autonomous driving scene, the driving command is unknown.",
        ]
    )

    used_action_channel_ids: List[int] = field(default_factory=lambda: [0, 1, 2])
    inverse_used_action_channel_ids: List[int] = field(default_factory=list)

    action_norm_method: str = "quantiles"
    norm_stat: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "q01": [-0.01582209, -1.76743735, -0.18790549],
            "q99": [6.15755262, 2.46998216, 0.18174374],
        }
    )

    enable_text_encoder_cache: bool = True

    def __post_init__(self) -> None:
        self.action_per_frame = self.action_chunk_steps

        inverse_used_action_channel_ids = [len(self.used_action_channel_ids)] * self.action_dim
        for index, channel_id in enumerate(self.used_action_channel_ids):
            inverse_used_action_channel_ids[channel_id] = index
        self.inverse_used_action_channel_ids = inverse_used_action_channel_ids
