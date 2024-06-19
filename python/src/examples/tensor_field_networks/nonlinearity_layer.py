
import math
import torch
from torch.nn import Module
from torch_geometric.data import Data
from typing import Any, Callable, Generic, List, TypeVar

from gelib import SO3vecArr

from src.examples.common.point_cloud import PointCloud

class TfnNonlinearityLayer(Module):
    """
    Applies an element-wise nonlinearity of type |nonlinearity_fn| to all
    entries of input tensor.
    """
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
    def reset_parameters(self):
        if self.real_bias_ == None:
            return
        
        self.real_bias_.reset_parameters()
        self.imaginary_bias_.reset_parameters()

    def createBiases(self, point_cloud : PointCloud):
        size = list(point_cloud.size())[0:-2]
        size.append(point_cloud.max_l + 1)
        size = tuple(size)

        self.real_bias_ = torch.zeros(size)
        self.imaginary_bias_ = torch.zeros(size)

    def forward(self, data : PointCloud):
        assert isinstance(data, PointCloud)

        # Calculate all V values.
        results = [ self.applyZeroNonlinearity(data.part(0)) ]
        for i in range(1, data.max_l() + 1):
            part = data.part(i)
            results.append(self.calculateNorm(part))
        v_tensor = torch.stack(results)

        # Add bias term.
        v_tensor = v_tensor + self.getBiasMatrix(data)

        # Apply nonlinearity.
        v_tensor = self.nonlinearity_(v_tensor)

        # Turn it back into another PointCloud.
        assert data.max_l() + 1 == v_tensor.size()[-1]
        results = [ v_tensor[...,0] ]
        for i in range(1, v_tensor.size()[-1]):
            part = data.part(i)
            factor = results[...,i].expand_as(part)
            results.append(part * factor)
        
        return data.CloneWithNewValue(SO3vecArr(results))
    
    def calculateNorm(self, tensor : torch.Tensor):
        return torch.norm(tensor, dim = -2)
    
    def getBiasMatrix(self, point_cloud):
        # Create biases, as the size isn't known until runtime.
        if self.real_bias_ == None:
            assert self.imaginary_bias_ == None
            self.real_bias_ = self.createBiases(point_cloud)

        bias = torch.complex(self.real_bias_, self.imaginary_bias_)

        new_size = list(bias.size())
        new_size[-1] = point_cloud.size()[-1]
        return bias.unsqueeze(-1).expand(new_size)
    
    def complex_relu(z : torch.Tensor):
        """Applies ReLU to the real and imaginary parts separately."""
        return torch.complex(torch.relu(z.real), torch.relu(z.imag))