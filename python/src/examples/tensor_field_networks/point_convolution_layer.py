import math
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from typing import List, Tuple

from ...gelib import SO3part
from ...gelib import SO3partArr

from src.examples.common.pyg_helper import \
    flattenForPygPropegate, undoFlattenForPygPropegate, reshapeInputForPyg, \
        undoReshapeInputForPyg

class PointConvolutionLayer(MessagePassing):
    def __init__(self,
                 channels : int,
                 l_filter: int):
        super().__init__(aggr='add')  # "Add" aggregation

        self.channels_ : int = channels
        self.l_filter_ : int = l_filter

        # Create a list to store MLPs for each output channel to handle learning
        # the radial function for each channel, as a function of the distance
        # between 2 points.
        kNumHiddenLayerNodes = 3
        self.r_mlps = torch.nn.ModuleList([])
        for channel in range(self.channels_):
            channel_list = torch.nn.ModuleList([])
            for l in range(2 * self.l_filter_ + 1):
                channel_list.append(Sequential(
                    Linear(1, kNumHiddenLayerNodes), 
                    ReLU(),
                    Linear(kNumHiddenLayerNodes, 1)))
            self.r_mlps.append(channel_list)

        self.reset_parameters()

    def reset_parameters(self):
        pass

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

        # edge_index has shape [2, |E|].
        i_arr, j_arr = edge_index

        # Get the spherical harmonics for each channel.
        sh_per_channel = self.getFilterValue(i_arr, j_arr)
        assert isinstance(sh_per_channel, SO3partArr)
        
        # Add dimensions to spherical harmonics to match x_j.
        assert sh_per_channel.dim() <= x_j.dim()
        while sh_per_channel.dim() < x_j.dim():
            sh_per_channel = sh_per_channel.unsqueeze(0)
        sh_per_channel = sh_per_channel.expand_as(x_j)

        # Calculate CG product.
        cg_products = sh_per_channel.DiagCGproduct(x_j, self.l_filter_)
        
        assert cg_products.size()[-1] == x_j.size()[-1], cg_products.size()

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
    
    def getFilterValue(self, i_arr, j_arr) -> SO3partArr:
        # Get a copy of the spherical harmonics for each channel
        sh_per_channel = self.getSphericalHarmonicsForFilter(i_arr, j_arr)
        
        # Apply the learned radial function to distances between i and j arrays.
        mlp_vals = self.getRValues(i_arr, j_arr)
        assert mlp_vals.dim() == sh_per_channel.dim() and \
               mlp_vals.size()[0:-4] == sh_per_channel.size()[0:-4] and \
               mlp_vals.size()[-2:-1] == sh_per_channel.size()[-2:-1], \
            "{0} vs {1}".format(mlp_vals.size(), sh_per_channel.size())
        sh_per_channel = sh_per_channel.expand_as(mlp_vals)

        # Combine the two and return it.
        sh_per_channel = sh_per_channel * mlp_vals
        assert sh_per_channel.dim() == 3, sh_per_channel.size()
        
        return sh_per_channel
    
    def getRValues(self, i_arr, j_arr):
        # Add an extra dimension so all MLPs can be run in parallel
        distance = self.getDistance(i_arr, j_arr).unsqueeze(-1)
        # temp = torch.stack([mlp(distance) for mlp in self.r_mlps[0]], dim=-1).size()
        # assert False, temp

        # Calculate all MLP results.
        mlp_results = torch.stack([
            torch.stack([mlp(distance) for mlp in mlp_list], dim=-1) \
                for mlp_list in self.r_mlps
        ])

        # Get rid of the extra dimension added above.
        assert mlp_results.size()[-2] == 1, mlp_results.size()
        mlp_results = mlp_results.squeeze(-2).transpose(1,2)
        assert mlp_results.dim() == 3, mlp_results.size()

        return mlp_results
        
    # Gets the spherical harmonics assocaited with the self.l_filter for this
    # layer.
    # NOTE: Does NOT depend on channel number. In the original TFN paper, this
    # is the Y_m^{(l_f)}(\hat{r}) spherical harmonic for the filter
    def getSphericalHarmonicsForFilter(self, i_arr, j_arr) -> SO3part:
        # Get the vector
        i_pos = self.point_positions_[i_arr]
        j_pos = self.point_positions_[j_arr]
        distance = self.getDistance(i_arr, j_arr)
        assert len(i_pos) == len(j_pos)
        assert len(i_pos) == len(distance)

        vector = (i_pos - j_pos)
        for k in range(len(vector)):
            vector[k] /= distance[k]

        # Extend in a new dimension by copying once per channel
        # spharm expects (batch_size, 3 (dimensions), N)
        assert vector.size()[-1] == 3
        vector = vector.unsqueeze(-1).transpose(0, -1)

        # Return the Spherical Harmonic associated with it
        return SO3partArr.spharm(self.l_filter_, vector)
    
    def getDistance(self, i, j):
        if isinstance(i, int):
            assert isinstance(j, int)
            return self.point_distances_[i,j]
        
        assert i.size() == j.size()
        if i.dim() == 0:
            return self.getDistance(i.item(), j.item())
        
        return torch.tensor(
            [self.point_distances_[i[k],j[k]] for k in range(len(i))])