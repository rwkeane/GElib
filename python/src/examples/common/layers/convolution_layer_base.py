import torch
from torch_geometric.nn import MessagePassing

from examples.common.layers.convolution_calculator import ConvolutionCalculator
from src.examples.common.point_cloud import PointCloud

# TODO: MessageP
class ConvolutionLayerBase(ConvolutionCalculator, MessagePassing):
    """
    Abstract class for all shared functionality associated with a GNN layer that
    performs a convolution operation. Based on the Pytorch Geometric
    MessagePassing primitives.
    """
    def __init__(self, channels : int, l_filter : int, l_max : int):
        # Calls both parent class ctors in order of MRO:
        # 1. ConvolutionCalculator with |channels|, |l_filter|, so these
        #    parameters are removed from parameter pack.
        # 2. MessagePassing with "Add" aggregation
        #
        # NOTE: Initializing torch.nn.Module from ConvolutionCalculator rather
        # than from MessagePassing doesn't SEEM to have any negative side
        # effects, but if weirdness happens later this may be why.
        super().__init__(aggr = 'add',
                         channels = channels,
                         l_filter = l_filter,
                         l_max = l_max)

        self.l_filter_ = l_filter
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: Ensure this works as expected.
        super(ConvolutionCalculator, self).reset_parameters()
        MessagePassing.reset_parameters(self)

    def forward(self, point_cloud : PointCloud):
        assert isinstance(point_cloud, PointCloud)

        # x of shape [num_nodes, channel_count, 2l_in + 1, N atoms]
        point_cloud = point_cloud.ToPygPropegationFormat()
        out = self.propagate(edge_index = point_cloud.edge_list(),
                             x = point_cloud)
        assert isinstance(out, PointCloud), type(out)
        return out.FromPygPropegationFormat()

    # Constructs message from node j to node i, which is then aggregated as
    # specified in ctor.
    def message(self,
                x_j : PointCloud,
                edge_index : torch.Tensor):
        # Reshape the underlying x_j so that its more easily understood by
        # later code.
        assert x_j.dim() == 2, x_j.size()
        x_j = x_j.FromPygPropegationFormat()

        # Call into ConvolutionCalculator for the actual calculations.
        cg_products = super().forward(x_j)
        assert cg_products != None and isinstance(cg_products, PointCloud)

        # Convert the result back into the format PyG expects, and return it.
        return cg_products.ToPygPropegationFormat()