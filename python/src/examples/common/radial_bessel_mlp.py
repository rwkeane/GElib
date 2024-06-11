from typing import Callable
import torch
from torch.nn import Linear, Module, ModuleList, Parameter

kDefaultMlpDepth = 3
kDefaultFanSize = 1
class RadialBesselMlp(Module):
    def __init__(self,
                 r_c : int,
                 num_basis : int,
                 p : int,
                 trainable_embedding : bool,
                 nonlinearity_fn : Callable[[torch.Tensor], torch.Tensor],
                 mlp_depth : int = kDefaultMlpDepth,
                 fan_size : int = kDefaultFanSize):
        layers = [ 
            RadialBesselFunction(num_basis, r_c, p, trainable_embedding) ]
        kUseBias = True
        for _ in range(mlp_depth):
            layers.append(Linear(fan_size, fan_size, bias = kUseBias))
            layers.append(NonlinearityLayer(nonlinearity_fn))
        layers.append(Linear(fan_size, fan_size, bias = kUseBias))

        layers = layers.reverse()
        self.layers_ = ModuleList(layers)

        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.layers_:
            layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers_:
            x = layer.forward(x)
        return x

class NonlinearityLayer(Module):
    def __init__(self,
                 nonlinearity_fn : Callable[[torch.Tensor], torch.Tensor]):
        self.nonlinearity_ = nonlinearity_fn

    def reset_parameters():
        pass

    def forward(self, x):
        return self.nonlinearity_(x)
    
class RadialBesselFunction(Module):
    def __init__(self, num_bases : int, r_c : int, p : float, trainable : bool):
        self.r_c_ = r_c
        self.p_ = float(p)

        weights = (
            torch.linspace(start = 1.0,
                           end = num_bases,
                           steps = num_bases) * torch.pi
        )
        if trainable:
            self.weights_ = Parameter(weights)
        else:
            self.register_buffer("weights", weights)

        self.reset_parameters()
    
    def reset_parameters(self):
        pass

    def forward(self, x : torch.Tensor):
        assert isinstance(x, torch.Tensor)

        # TODO: Replace x with x.unsqueeze(-1) on 2 below lines?
        first_factor = 2 / self.r_c_
        second_factor = torch.sin(self.weights_ * x / self.r_c_) / x

        return first_factor * second_factor * self.getPolynomial(x)
    
    def getPolynomial(self, distance : torch.Tensor):
        """
        Polynomial cutoff, as per DimeNet: https://arxiv.org/abs/2003.03123
        """
        out = 1.0 \
            - (((self.p_ + 1.0) * (self.p_ + 2.0) / 2.0) \
               * torch.pow(distance, self.p_))\
            + (self.p_ * (self.p_ + 2.0) \
               * torch.pow(distance, self.p_ + 1.0)) \
            - ((self.p_ * (self.p_ + 1.0) / 2) \
               * torch.pow(distance, self.p_ + 2.0))

        cutoff = ((distance / self.r_c_) < 1.0).int()
        return out * cutoff