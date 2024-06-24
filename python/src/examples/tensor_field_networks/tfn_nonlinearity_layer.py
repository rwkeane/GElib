import torch
from torch.nn import Module
from typing import Any, Callable

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
        self.real_bias_ : torch.Tensor = None
        self.imaginary_bias_ : torch.Tensor = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.real_bias_ == None:
            return
        
        self.real_bias_ = torch.zeros(self.real_bias_.size(),
                                      requires_grad = True)
        self.imaginary_bias_ = torch.zeros(self.imaginary_bias_.size(),
                                           requires_grad = True)

    def createBiases(self, point_cloud : PointCloud):
        size = list(point_cloud.size())[:-1]
        size[-1] = point_cloud.max_l() + 1
        size = tuple(size)

        self.real_bias_ = torch.zeros(size, requires_grad = True)
        self.imaginary_bias_ = torch.zeros(size, requires_grad = True)

        assert self.real_bias_.size() == self.imaginary_bias_.size()
        assert self.real_bias_.dim() == point_cloud.dim() - 1

    def forward(self, data : PointCloud):
        assert isinstance(data, PointCloud)

        # Calculate all V values.
        results = [ self.getLZeroPart(data.part(0)) ]
        for i in range(data.max_l()):
            part = data.part(i)
            results.append(self.calculateNorm(part))
        v_tensor = torch.stack(results, dim = -2)
        assert v_tensor.dim() == data.dim(), v_tensor.size()

        # Add bias term.
        v_tensor = v_tensor + self.getBiasMatrix(data)

        # Apply nonlinearity.
        v_tensor = self.nonlinearity_(v_tensor)

        # Turn it back into another PointCloud.
        assert v_tensor.size()[-2] == data.max_l() + 1
        assert v_tensor.dim() == data.dim(), v_tensor.size()
        results = [ v_tensor[...,0,:].unsqueeze(-2) ]
        for i in range(1, v_tensor.size()[-2]):
            part = data.part(i)
            factor = v_tensor[...,i,:].unsqueeze(-2).expand_as(part)
            results.append(part * factor)
        
        assert len(results) == data.max_l() + 1
        new_cloud = data.CloneWithNewValue(results)
        return new_cloud
    
    def calculateNorm(self, tensor : torch.Tensor):
        return torch.norm(tensor, dim = -2)
    
    def getBiasMatrix(self, point_cloud : PointCloud):
        assert isinstance(point_cloud, PointCloud)

        # Create biases, as the size isn't known until runtime.
        if self.real_bias_ == None:
            assert self.imaginary_bias_ == None
            self.createBiases(point_cloud)

        bias = torch.complex(self.real_bias_, self.imaginary_bias_)

        # Expand it, with the new -2 dimension as the number of rotation orders.
        new_size = list(bias.size())
        new_size.append(point_cloud.size()[-1])
        bias = bias.unsqueeze(-1).expand(new_size)
        assert bias.size()[-2] == point_cloud.max_l() + 1, bias.size()
        return bias
    
    def getLZeroPart(self, data : torch.Tensor):
        assert isinstance(data, torch.Tensor), type(data)
        assert data.size()[-2] == 1, "Must be the l=0 rotation order"
        
        return data.squeeze(-2)
    
    def complex_relu(z : torch.Tensor):
        """Applies ReLU to the real and imaginary parts separately."""
        assert isinstance(z, torch.Tensor)

        return torch.complex(torch.relu(z.real), torch.relu(z.imag))