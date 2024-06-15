import torch
from functools import partial
from torch.nn import Linear, Module, ModuleList

from ...gelib import SO3vecArr

from src.examples.common.radial_bessel_mlp_stack import RadialBesselMlpStack
from src.examples.common.convolution_calculator import ConvolutionCalculator

class APart(ConvolutionCalculator):
    """
    Represents the A function, as defined in MACE, with |channelse| channels and
    max l-value |l_filter|.

    TODO: Write a special case for the first layer, where computation is
    significantly easier.
    """
    def __init__(self, channels : int, l_filter: int):
        # Initialize ConvolutionCalculator.
        super(ConvolutionCalculator, self).__init__(channels, l_filter)

        # Weights for point representations.
        self.representation_weights_ = ModuleList([])
        for _ in range(l_filter):
            self.representation_weights_.append(Linear(channels, channels))

        # TODO: Check these values against the paper.
        kRadialCutoff = 1.0
        kNumBasis = 8
        kPValue = 5
        kTrainable = False
        self.r_mlps_ = RadialBesselMlpStack(
            channels, l_filter, kRadialCutoff, kNumBasis, kPValue, kTrainable,
            torch.relu)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        self.r_mlps_.reset_parameters()

        for module in self.representation_weights_:
            assert isinstance(module, Linear)
            module.reset_parameters()

    def forward(self, x_j : SO3vecArr, edge_index : torch.Tensor):
        return super().forward(x_j, edge_index)

    # ConvolutionCalculate abstract method.
    def calculateRadialValues(
            self, point_distances : torch.Tensor) -> torch.Tensor:
        # Add an extra dimension so all MLPs can be run in parallel
        distance = point_distances.unsqueeze(-1)

        # Calculate all MLP results.
        mlp_results = self.r_mlps_.forward(distance)
        
        return mlp_results
  
    # ConvolutionCalculator virtual method.
    def getPointRepresentation(self, x_j : torch.Tensor) -> torch.Tensor:
        # x_j represents the "source" nodes, and is of shape
        # [<some_num_nodes>, ..., channel_count, 2l_in + 1, N atoms].
        assert x_j.size()[1] == len(self.representation_weights_)

        # Resize to [X, l_index, channel].
        transposed_size = x_j.transpose(-3, -1).size()
        x_j = x_j.reshape(-1, transposed_size[-2], transposed_size[-1])

        # Apply linear layers.
        result = torch.stack(
            [self.representation_weights_[i].forward(x_j[:,i,:]) \
                for i in range(len(self.representation_weights_))],
            dim = -2
        )

        # Un-swap and un-resize.
        result = result.reshape(transposed_size).transpose(-3, -1)

        return result
