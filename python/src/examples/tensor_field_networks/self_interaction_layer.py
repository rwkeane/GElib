import math
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from typing import Any, Callable, Generic, List, TypeVar

from ...gelib import SO3vecArr

from src.examples.common.point_cloud import PointCloud

class SelfInteractionLayer(Module):
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 l_in : int):
        super().__init__()
        assert l_in >= 0

        self.l_in_ = l_in
        self.in_channels_ = in_channels
        self.out_channels_ = out_channels

        self.l_filters_ = torch.nn.ModuleList(
            [ Linear(in_channels, out_channels, bias = (i == 0),
                     dtype=torch.cfloat) for i in range(2 * l_in + 1) ])

        self.reset_parameters()

    def reset_parameters(self):
        for filter in self.l_filters_:
            assert isinstance(filter, Linear)
            filter.reset_parameters()

    def forward(self, x : PointCloud):
        # NOTE: this layer assumes -3 is the channel dim, of 4+.
        # x of shape [batch, channel_count, 2l_in + 1, N atoms]
        assert isinstance(x, PointCloud)
        assert x.size()[-3] == self.in_channels_

        # New order [N atoms, batch, channel, 2l_in + 1]
        order = list(range(-1, x.dim() - 1, 1))
        permuted : PointCloud = x.permute(order)
        permuted_size = permuted.size()
        x_reshaped = permuted.view(-1, permuted_size[-2], permuted_size[-1])

        # Sum across channels for each l index (do NOT mix across l vals).
        y_reshaped = SO3vecArr(torch.stack(
            [self.l_filters_[i].forward(x_reshaped[...,i,]) \
                for i in range(permuted_size[-1])], -1))
        y_reshaped = x.CloneWithNewValue(y_reshaped)
        
        # Validate it worked.
        if __debug__:
            expected_new_size = list(x_reshaped.size())
            expected_new_size[-2] = self.out_channels_
            assert list(y_reshaped.size()) == expected_new_size, \
                "{0} vs {1}".format(y_reshaped.size(), expected_new_size)
        
        # Undo the reshaping.
        new_size = list(permuted.size())
        new_size[-2] = self.out_channels_
        y : PointCloud = y_reshaped.view(new_size)

        # Undo re-ordering.
        order = list(range(1, y.dim(), 1))
        order.append(0)
        order = tuple(order)
        y = y.permute(order)

        assert y.size()[-3] == self.out_channels_
        return y