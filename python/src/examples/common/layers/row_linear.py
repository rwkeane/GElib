from typing import List
from functools import partial

import torch
from torch.nn import Linear, Module, ModuleList
from gelib import SO3vecArr

from src.examples.common.point_cloud import PointCloud
from src.examples.common.impl.point_cloud_base import PointCloudBase
from examples.common.impl.util.internal_caller import InternalCaller

class ROWLinear(InternalCaller, Module):
    """
    A Rotation-Order-Wise linear layer, where a separate linear layer with 
    |in_features| and |out_features| is applied across each rotation order (each
    part of the underlying SO3vecArr).
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 max_l : int,
                 dim : int = -1,
                 biases : List[bool] = None,
                 device = None,
                 dtype = torch.cfloat) -> None:
        super().__init__()
        if biases == None:
            biases = [ False for i in range(max_l + 1)]

        assert len(biases) == max_l + 1

        self.in_features_ = in_features
        self.out_features_ = out_features
        self.dim_ = dim
        self.max_l_ = max_l
        self.linears_ = ModuleList([ Linear(in_features,
                                   out_features,
                                   biases[i],
                                   dtype = dtype,
                                   device = device) for i in range(max_l + 1) ])

        self.reset_parameters()

    def reset_parameters(self):
        for filter in self.linears_:
            assert isinstance(filter, Linear)
            filter.reset_parameters()

    def forward(self, point_cloud : PointCloud):
        # NOTE: this layer assumes -3 is the channel dim, of 4+.
        # x of shape [batch, channel_count, 2l_in + 1, N atoms]
        assert isinstance(point_cloud, PointCloudBase)
        assert point_cloud.max_l() == self.max_l_, "{0} vs {1}".format(point_cloud.max_l(), self.max_l_)

        # Operate on the correct dimension.
        if self.dim_ != -1:
            point_cloud = point_cloud.transpose(self.dim_, -1)
            
        # Reshape to a form linear layers can process.
        all_sizes = point_cloud.allSizes()
        point_cloud = point_cloud.view(-1, (all_sizes[0])[-1])

        # Feed reshaped data in to the linear layer.
        results = [self.linears_[i].forward(point_cloud.part(i)) \
                        for i in range(point_cloud.max_l() + 1)]
        point_cloud : PointCloudBase = point_cloud.CloneWithNewValue(results)

        # Reshape the point cloud back to the original shape, with modified last
        # dimension if the linear layers changed it.
        if self.in_features_ != self.out_features_:
          for i in range(len(all_sizes)):
              size = list(all_sizes[i])
              size[-1] == self.out_features_
              all_sizes[i] = tuple(size)

        point_cloud = point_cloud.allViews(all_sizes)

        # Swap dimensions back if they were swapped earlier.
        if self.dim_ != -1:
            point_cloud = point_cloud.transpose(self.dim_, -1)
            
        return point_cloud