import torch
from functools import partial
from torch.nn import Linear, Module, Parameter, ReLU, Sequential

from src.examples.common.radial_bessel_mlp_stack import RadialBesselMlpStack
from src.examples.common.convolution_layer_base import ConvolutionLayerBase
from src.examples.nequip.nequip_utils import kPositive, kNegative

class NequipConvolutionLayer(ConvolutionLayerBase):
    def __init__(self,
                 channels : int,
                 l_filter : int,
                 parity : int):
        super().__init__(channels, l_filter)

        assert parity == kPositive or parity == kNegative
        self.filter_parity_ = parity

        # TODO: Verify below values.
        kRadialCutoff = 1.0
        kNumBasis = 8       # From original source code.
        kPValue = 6         # From paper.
        kTrainable = True   # From paper
        self.r_positive_mlps_ = RadialBesselMlpStack(
            channels, l_filter, kRadialCutoff, kNumBasis, kPValue, kTrainable)
        self.r_negative_mlps_ = RadialBesselMlpStack(
            channels, l_filter, kRadialCutoff, kNumBasis, kPValue, kTrainable)

        self.reset_parameters()

    def reset_parameters(self):
        self.r_positive_mlps_.reset_parameters()
        self.r_negative_mlps_.reset_parameters()
    
    def calculateRadialValues(self, point_distances : torch.Tensor):
        # Add an extra dimension so all MLPs can be run in parallel
        distance = point_distances.unsqueeze(-1)

        # Calculate all MLP results.
        positive_mlp_results = \
            self.r_positive_mlps_.forward(distance).unsqueeze(-4)
        negative_mlp_results = \
            self.r_negative_mlps_.forward(distance).unsqueeze(-4)

        if self.filter_parity_ == kPositive:
            mlp_results = torch.stack(
                [ positive_mlp_results, negative_mlp_results ], dim = -4)
        else:
            assert self.filter_parity_ == kNegative
            mlp_results = torch.stack(
                [ negative_mlp_results, positive_mlp_results ], dim = -4)

        return mlp_results