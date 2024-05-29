import math
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from typing import Any, Callable, Generic, List, TypeVar
from torch_geometric.data import Data

from ...gelib import SO3part
from ...gelib import SO3partArr

from src.examples.tensor_field_networks.channel_mapper import ChannelMapper

class SelfInteractionLayer(Module):
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 l_in : int):
        super().__init__()
        assert l_in >= 0

        self.l_in_ = l_in

        self.l_filters_ = torch.nn.ModuleList(
            [Linear(in_channels, out_channels, bias = (i == 0)) for i in \
                range(2 * l_in + 1)])

        self.reset_parameters()

    def reset_parameters(self):
        for filter in self.l_filters_:
            filter.reset_parameters()

    def forward(self, data : Data):
        # x of shape [num_nodes, 2l_in + 1, channel_count]
        x = data.x
        assert isinstance(x, SO3partArr)

        # Sum across channels
        data.x = SO3partArr(torch.stack(
            [self.l_filters_[i].forward(x[:,i,:]) for i in range(self.l_in_)]))

        return data