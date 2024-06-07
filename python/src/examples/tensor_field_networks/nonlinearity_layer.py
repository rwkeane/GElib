
import math
import torch
from torch.nn import Module
from torch_geometric.data import Data
from typing import Any, Callable, Generic, List, TypeVar

from ...gelib import SO3partArr

class TfnNonlinearityLayer(Module):
    def __init__(
            self,
            channels : int,
            l_in : int,
            nonlinearity_fn : Callable[[torch.Tensor], torch.Tensor] = None):
        super().__init__()

        if nonlinearity_fn == None:
            nonlinearity_fn = TfnNonlinearityLayer.complex_relu

        self.channels_ = channels 
        self.l_in_ = l_in
        self.nonlinearity_ = nonlinearity_fn

        self.reset_parameters()

    def reset_parameters(self):
        l_dim = 2 * self.l_in_ + 1
        self.real_bias_ = \
            torch.zeros((l_dim, self.channels_), requires_grad = True)
        self.imaginary_bias_ = \
             torch.zeros((l_dim, self.channels_), requires_grad = True)
        assert self.real_bias_.size() == self.imaginary_bias_.size()

    def forward(self, data : Data):
        # x of shape [batch, channel_count, 2l_in + 1, N atoms]
        x = data.x
        assert x.size()[-3] == self.channels_

        data.x = torch.cat([
            self.applyZeroNonlinearity(x[...,0,:]).unsqueeze(-2),  # l = 0
            self.applyNonZeroNonlinearity(x[...,1:,:])  # l > 0
        ], dim = -2)
        return data
    
    def applyZeroNonlinearity(self, l_zero : torch.Tensor):
        assert l_zero.dim() >= 2

        bias = torch.complex(self.real_bias_[0,:], self.imaginary_bias_[0,:])
        while bias.dim() < l_zero.dim():
            bias = bias.unsqueeze(0)
        bias = bias.expand_as(l_zero)
        return self.nonlinearity_(l_zero + bias)
    
    def applyNonZeroNonlinearity(self, l_all : torch.Tensor):
        assert l_all.dim() >= 3

        normed = torch.norm(l_all, dim = -2).unsqueeze(-2).expand_as(l_all)
        bias = torch.complex(self.real_bias_[1:,:], self.imaginary_bias_[1:,:])
        while bias.dim() < normed.dim():
            bias = bias.unsqueeze(0)
        bias = bias.expand_as(normed)

        return self.nonlinearity_(normed + bias) * l_all
    
    def complex_relu(z : torch.Tensor):
        """Applies ReLU to the real and imaginary parts separately."""
        return torch.complex(torch.relu(z.real), torch.relu(z.imag))