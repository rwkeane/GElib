from typing import Callable
import torch
from torch.nn import Linear, Module, ModuleList, Parameter

# Default number of fully-connected layers to use in a RadialBesselMlp.
kDefaultMlpDepth = 1

# Default number of input / output / hidden nodes to use in a RadialBesselMlp.
kDefaultFanSize = 1

class RadialBesselMlp(Module):
    """
    An MLP of depth 2 * (|mlp_depth| + 1) with fan in size, fan out of size, and
    hidden layer node count |fan_size|.
    
    The first layer of the MLP is a basis embedding of the interatomic distance,
    calculated using radial bessel functions with polynomial cutoff
    (using p=|p|, |num_basis| bases, and |r_c| cutoff value). This is followed
    by a nonlinearity of type |nonlinearity_fn|, and then |mlp_depth| pairs of
    (nonlinearity, fully connected) layers.
    """
    def __init__(self,
                 r_c : int,
                 num_basis : int,
                 p : int,
                 trainable_embedding : bool,
                 nonlinearity_fn : Callable[[torch.Tensor], torch.Tensor],
                 mlp_depth : int = kDefaultMlpDepth,
                 fan_size : int = kDefaultFanSize):
        super().__init__()

        # Specify if the MLP should use biases.
        # 
        # TODO: This may need to be a ctor parameter?
        kUseBias = True

        # There must always be at least one layer, to apply the radial bessel
        # function.
        layers = [ 
            RadialBesselFunction(num_basis, r_c, p, trainable_embedding) ]
        layers.append(Linear(num_basis, fan_size, bias = kUseBias))

        # Add additional fully connected layers and nonlinearities based on
        # ctor parameter |mlp_depth|.
        for _ in range(mlp_depth):
            layers.append(NonlinearityLayer(nonlinearity_fn))
            layers.append(Linear(fan_size, fan_size, bias = kUseBias))

        # Create the layers, with the RadialBesselFunction being the FIRST
        # applied.
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
    """
    MLP layer to perform nonlinearity specified by |nonlinearity_fn|.
    """
    def __init__(self,
                 nonlinearity_fn : Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()

        assert nonlinearity_fn != None
        self.nonlinearity_ = nonlinearity_fn

    def reset_parameters(self):
        pass

    def forward(self, x):
        return self.nonlinearity_(x)
    
class RadialBesselFunction(Module):
    """
    Calculates the Radial Bessel function, with polynomial cuttoff.
    """
    def __init__(self, num_bases : int, r_c : int, p : float, trainable : bool):
        super().__init__()

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
            self.register_buffer("weights_", weights)

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