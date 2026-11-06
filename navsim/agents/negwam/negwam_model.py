import os
from typing import Dict
import torch
import logging
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.negwam.negwam_config import NegwamConfig
from navsim.agents.negwam.negwam_backbone import NegwamBackbone
from navsim.agents.negwam.modules.navsim_model import WanTransformer3DNavSimModel




logger = logging.getLogger(__name__)

class NegwamModel(nn.Module):
    """Negwam model"""

    def __init__(self, config: NegwamConfig):
        """
        Initializes Negwam model
        :param trajectory_sampling: trajectory sampling specification
        :param config: global config of Negwam model
        """
        super().__init__()  
        self.config = config
        self.backbone = NegwamBackbone(config)

        self.transformer = WanTransformer3DNavSimModel.from_pretrained(
            config.checkpoint_path,
            torch_dtype=torch.float32,
        ).to("cpu")  

        if not config.enable_compat_head:
           
            self.transformer.compat_head.requires_grad_(False)


    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Negwam model
        :param features: dictionary of features
        :return: dictionary of predictions
        """
        input_dict = self.backbone(features)
        
        if self.training:
            return self.forward_train(input_dict)
        else:
            # print("Inference mode not implemented yet")
            return self.forward_val(input_dict)
    

    def forward_train(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Negwam model in training mode
        :param input_dict: dictionary of input features
        :return: dictionary of predictions
        """
        self.transformer.train()
        self.transformer.requires_grad_(True)
        if not self.config.enable_compat_head:
            self.transformer.compat_head.requires_grad_(False)   

        output = self.transformer(input_dict, train_mode=True)

        return input_dict, output


    def forward_val(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Negwam model in inference mode
        :param input_dict: dictionary of input features
        :return: dictionary of predictions
        """
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        output = self.transformer(input_dict, train_mode=True)

        return input_dict, output


    
    def configure_gradient_clipping(self, gradient_clip_val):
        """
        Configure the gradient clipping
        """
        self.transformer.clip_grad_norm_(gradient_clip_val)



    
    


