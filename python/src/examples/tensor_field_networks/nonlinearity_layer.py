
import math
import torch
from torch.nn import Module
from torch_geometric.data import Data
from typing import Any, Callable, Generic, List, TypeVar

from ...gelib import SO3partArr

class TfnNonlinearityLayer(Module):
    def __init__(self,
                 channels : int,
                 l_in : int,
                 nonlinearity_fn = torch.relu):
        self.channels_ : int = channels 
        self.l_in_ = l_in
        self.nonlinearity = nonlinearity_fn

        self.learnable_bias_ = torch.zeros((2 * l_in + 1, channels),
                                           requires_grad = True)

        self.reset_parameters()

    def reset_parameters(self):
        for filter in self.l_filters_:
            filter.reset_parameters()

    def forward(self, data : Data):
        # x of shape [num_nodes, 2l_in + 1, channel_count]
        x = data.x

        data.x[:,0,:] = self.applyZeroNonlinearity(x[:,0,:])
        data.x[:,1:,:] = self.applyNonZeroNonlinearity(x[:,1:,:])
        return data
    
    def applyZeroNonlinearity(self, l_zero : torch.Tensor):
        assert l_zero.dim() == 2

        bias = self.learnable_bias_[0,:].expand_as(l_zero)
        return self.nonlinearity(l_zero + bias)
    
    def applyNonZeroNonlinearity(self, l_all : torch.Tensor):
        assert l_all.dim() == 3
        normed = torch.norm(l_all, dim = -2)
        bias = self.learnable_bias_[1:,:].expand_as(normed)

        added = normed + bias
        added = self.nonlinearity(added)

        return added.expand_as(l_all) * l_all