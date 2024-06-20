from typing import Optional

import torch
import torch.nn as nn

from src.examples.nequip.nequip_utils import kNegative, kPositive
from src.examples.nequip.nequip_convolution_layer import NequipConvolutionLayer
from src.examples.tensor_field_networks.self_interaction_layer import \
    SelfInteractionLayer
from src.examples.common.point_cloud import PointCloud

class InteractionBlock(torch.nn.Module):
    """
    Defines the Interaction Block, consisting of:
    - Self Interaction Layer
    - Convolution Layer (which also includes concatination)
    - Self Interaction Layer
    - SiLU Nonlinearity
    """
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 l_max : int,
                 parity : int,
                 mid_channels : Optional[int] = None):
        super().__init__()

        assert parity == kNegative or parity == kPositive
        if mid_channels == None:
            mid_channels = in_channels

        self.first_interation_ = \
            SelfInteractionLayer(in_channels, mid_channels, l_max)
        self.convolution_ = NequipConvolutionLayer(mid_channels, l_max, parity)
        self.second_interaction_ = \
            SelfInteractionLayer(mid_channels, out_channels, l_max)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.first_interation_.reset_parameters()
        self.second_interaction_.reset_parameters()
        self.convolution_.reset_parameters()

    def forward(self, point_cloud : PointCloud):
        point_cloud = self.first_interation_.forward(point_cloud)
        point_cloud = self.convolution_.forward(point_cloud)
        point_cloud = self.second_interaction_.forward(point_cloud)

        assert False, "The below isnt equivarient! It must be wrong!"
        point_cloud = torch.nn.functional.silu(point_cloud)

        return data