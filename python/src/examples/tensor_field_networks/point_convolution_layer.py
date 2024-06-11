import torch
from functools import partial
from torch.nn import Linear, Module, Parameter, ReLU, Sequential

from src.examples.common.radial_bessel_mlp_stack import RadialBesselMlpStack
from src.examples.common.convolution_layer_base import ConvolutionLayerBase

class PointConvolutionLayer(ConvolutionLayerBase):
    def __init__(self,
                 channels : int,
                 l_filter: int):
        super().__init__(channels, l_filter)

        # Use rbmlp to allow better comparison to other models.
        kRadialCutoff = 1.0
        kNumBasis = 8
        kPValue = 6
        kTrainable = False
        self.r_mlps_ = RadialBesselMlpStack(
            channels, l_filter, kRadialCutoff, kNumBasis, kPValue, kTrainable)

        self.reset_parameters()

    def reset_parameters(self):
        self.r_mlps_.reset_parameters()
    
    def calculateRadialValues(self, point_distances : torch.Tensor):
        # Add an extra dimension so all MLPs can be run in parallel
        distance = point_distances.unsqueeze(-1)

        # Calculate all MLP results.
        mlp_results = self.r_mlps_.forward(distance)

        return mlp_results