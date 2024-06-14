import torch

from src.examples.common.radial_bessel_mlp_stack import RadialBesselMlpStack
from src.examples.common.convolution_layer_base import ConvolutionLayerBase
from src.examples.nequip.nequip_utils import kPositive, kNegative

class NequipConvolutionLayer(ConvolutionLayerBase):
    """
    Represents a convolution layer for the NequIP architecture, with |channels|
    channels, max l-value |l_filter|, and parity |parity|.
    """
    def __init__(self, channels : int, l_filter : int, parity : int):
        super().__init__(channels, l_filter)

        assert parity == kPositive or parity == kNegative
        self.filter_parity_ = parity

        # TODO: Verify below values.
        kRadialCutoff = 1.0
        kNumBasis = 8       # From original source code.
        kPValue = 6         # From paper.
        kTrainable = True   # From paper
        self.r_positive_mlps_ = RadialBesselMlpStack(
            channels, l_filter, kRadialCutoff, kNumBasis, kPValue, kTrainable,
            torch.selu)
        self.r_negative_mlps_ = RadialBesselMlpStack(
            channels, l_filter, kRadialCutoff, kNumBasis, kPValue, kTrainable,
            torch.selu)

        self.reset_parameters()

    def reset_parameters(self):
        self.r_positive_mlps_.reset_parameters()
        self.r_negative_mlps_.reset_parameters()
    
    def calculateRadialValues(self, point_distances : torch.Tensor):
        # Add an extra dimension so all MLPs can be run in parallel.
        distance = point_distances.unsqueeze(-1)

        # Calculate all MLP results.
        positive_mlp_results = \
            self.r_positive_mlps_.forward(distance).unsqueeze(-4)
        negative_mlp_results = \
            self.r_negative_mlps_.forward(distance).unsqueeze(-4)

        # Decide the order of these MLP results based on layer parity.
        if self.filter_parity_ == kPositive:
            mlp_results = torch.stack(
                [ positive_mlp_results, negative_mlp_results ], dim = -4)
        else:
            assert self.filter_parity_ == kNegative
            mlp_results = torch.stack(
                [ negative_mlp_results, positive_mlp_results ], dim = -4)

        return mlp_results
    
    def forward(self, data):
        data = super().forward(data)

        x : torch.Tensor = data.x

        # Get all (i, j pairs)
        edge_index : torch.Tensor = data.edge_index
        sources = torch.split(edge_index[0,:], 1, dim = -1)

        # Get the number of neighbors.
        sources = edge_index[0,:]
        num_nodes = x.size()[-1]
        counts = torch.histc(
            sources, bins = num_nodes, min = 0, max = num_nodes - 1)

        # Divide by sqrt(counts) when counts > 0.
        counts[counts == 0] = 1
        counts = torch.sqrt(counts)
        while counts.dim() < x.dim():
            counts.unsqueeze(0)
        counts.expand_as(x)

        # Store the updated values.
        data.x = x / counts

        return data