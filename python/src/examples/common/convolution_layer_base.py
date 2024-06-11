import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from ...gelib import SO3partArr

from src.examples.common.convolution_calculator import ConvolutionCalculator
from src.examples.common.pyg_helper import \
    flattenForPygPropegate, undoFlattenForPygPropegate, reshapeInputForPyg, \
        undoReshapeInputForPyg

class ConvolutionLayerBase(MessagePassing, ConvolutionCalculator):
    """
    Abstract class for all shared functionality associated with a GNN layer that
    performs a convolution operation. Based on the Pytorch Geometric
    MessagePassing primitives.
    """
    def __init__(self, channels : int, l_filter: int):
        # Calls MessagePassing init() with "Add" aggregation.
        super(MessagePassing, self).__init__(aggr='add')

        # Calls ConvolutionCalculator's init().
        super(ConvolutionCalculator, self).__init__(channels, l_filter)

        self.reset_parameters()

    def reset_parameters(self):
        super(MessagePassing, self).reset_parameters()
        super(ConvolutionCalculator, self).reset_parameters()

    def forward(self, data : Data):
        # x of shape [num_nodes, channel_count, 2l_in + 1, N atoms]
        x = data.x
        assert x.size()[-3] == self.channels_, \
            "{0} vs {1}".format(x.size()[-3], self.channels_)
        
        x = reshapeInputForPyg(x)
        edge_index = data.edge_index
        assert isinstance(x, SO3partArr)

        # Hack to overcome when the point distances may change.
        assert data.point_positions != None
        self.point_positions_ : torch.Tensor = data.point_positions
        assert self.point_positions_.size()[1] == 3, \
            self.point_positions_.size()

        assert data.point_distances != None
        self.point_distances_ : torch.Tensor = data.point_distances
        assert self.point_distances_.dim() == 2

        # Start propagating messages.
        #
        # NOTE: Hacky dimension thrashing needed to work with PyG.
        x, size = flattenForPygPropegate(x)
        out = self.propagate(edge_index, x = x, original_size = size[:])
        assert out.dim() == 2 and out.size()[0] == size[0], out.size()
        data.x = undoFlattenForPygPropegate(out, size)
        assert data.x.size() == size, data.x.size()
        data.x = undoReshapeInputForPyg(data.x)

        return data

    # Constructs message from node j to node i, which is then aggregated as
    # specified in ctor.
    def message(self, x_j : SO3partArr, edge_index : torch.Tensor,
                original_size : torch.Size):
        assert isinstance(original_size, torch.Size), type(original_size)

        # Hack the batch dimension of the SO3partArr because pyg does not
        # support multidimensional arrays of features, and instead expects that
        # batching is handled internally to the library.
        original_size = list(original_size)
        assert x_j.dim() == 2, x_j.size()
        input_size = x_j.size()
        original_size[0] = input_size[0]
        x_j = undoFlattenForPygPropegate(x_j, original_size)
        x_j = undoReshapeInputForPyg(x_j)

        # Call into ConvolutionCalculator for the actual calculations.
        cg_products = \
            super(ConvolutionCalculator, self).forward(x_j, edge_index)

        # Convert the result back into the format PyG expects, and return it. At
        # this point, the size should match that at input time, so validate
        # against that too.
        cg_products = reshapeInputForPyg(cg_products)
        assert list(cg_products.size()) == original_size, \
            "{0} vs {1}".format(list(cg_products.size()), original_size)
        flattened, new_size = flattenForPygPropegate(cg_products)
        assert flattened.size() == input_size, \
            "{0} vs {1}".format(flattened.size(), input_size)

        return flattened