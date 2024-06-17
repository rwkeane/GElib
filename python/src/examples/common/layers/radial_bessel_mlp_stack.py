from functools import partial
from typing import Callable, TypeVar, Generic
import torch

from examples.common.layers.mlp_stack import MlpStack
from examples.common.layers.radial_bessel_mlp import \
    RadialBesselMlp, kDefaultFanSize, kDefaultMlpDepth

class RadialBesselMlpStack(MlpStack):
    """
    Simple wrapper around MlpStack to abstract away the use of python
    functionals. See MlpStack and RadialBesselMlp for more info.
    """
    def __init__(self,
                 channels : int,
                 l_max : int,
                 r_c : int,
                 num_basis : int,
                 p : int,
                 trainable_embedding : bool,
                 nonlinearity_fn : Callable[[torch.Tensor], torch.Tensor],
                 mlp_depth : int = kDefaultMlpDepth,
                 fan_size : int = kDefaultFanSize):
        factory = partial(RadialBesselMlp,
                          r_c,
                          num_basis,
                          p,
                          trainable_embedding,
                          nonlinearity_fn,
                          mlp_depth,
                          fan_size)
        super().__init__(channels, l_max, factory)