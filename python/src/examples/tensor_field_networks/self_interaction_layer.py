import math
import torch

from gelib import SO3vecArr

from src.examples.common.layers.row_linear import ROWLinear
from src.examples.common.point_cloud import PointCloud

class SelfInteractionLayer(ROWLinear):
    def __init__(self, in_channels : int, out_channels : int, l_filter : int):
        filters = [i == 0 for i in range(l_filter + 1)]
        super().__init__(in_features = in_channels,
                         out_features = out_channels,
                         dim = -4,
                         max_l = l_filter,
                         biases = filters)
        
        self.in_channels_ = in_channels
        self.out_channels_ = out_channels

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x : PointCloud):
        # NOTE: this layer assumes -3 is the channel dim, of 4+.
        # x of shape [batch, ..., channel_count, 2l_in + 1, N atoms]
        assert isinstance(x, PointCloud)
        assert x.size()[-4] == self.in_channels_, x.size()
        
        y = super().forward(x)

        assert y.size()[-4] == self.out_channels_, y.size()
        return y