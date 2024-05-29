import math
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from typing import Any, Callable, Generic, List, TypeVar

from ...gelib import SO3part
from ...gelib import SO3partArr

from src.examples.tensor_field_networks.channel_mapper import ChannelMapper

class PointConvolutionLayer(MessagePassing):
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 l_filter: int,
                 point_distances : torch.Tensor):
        super().__init__(aggr='add')  # "Add" aggregation

        self.in_channels_ : int = in_channels
        self.out_channels_ : int = out_channels
        self.l_filter_ : int = l_filter
        self.point_distances_ = point_distances
        
        self.channel_map_ = \
            ChannelMapper(self.in_channels_, self.out_channels_, bias = False)

        # Create a list to store MLPs for each output channel to handle learning
        # the radial function for each channel, as a function of the distance
        # between 2 points.
        kNumHiddenLayerNodes = self.out_channels_
        self.r_mlps = torch.nn.ModuleList([
            Sequential(
                Linear(1, kNumHiddenLayerNodes), 
                ReLU(),
                Linear(kNumHiddenLayerNodes, 1)  # Output is a single value R_c
            ) for _ in range(self.in_channels)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        self.channel_map_.reset_parameters()
        for r_mlp in self.r_mlps:
            for layer in r_mlp:
                if isinstance(layer, Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, data : Data):
        # x of shape [num_nodes, 2l_in + 1, channel_count]
        x = data.x
        edge_index = data.edge_index
        assert isinstance(x, SO3partArr)

        # Hack to overcome when the point distances may change.
        if data.point_distances != None:
            self.point_distances_ = data.point_distances
        
        # Map from |in_channels| to |out_channels|.
        x = self.channel_map_.forward(x)

        # Start propagating messages.
        out = self.propagate(edge_index, x)
        data.x = out
        return data

    # Constructs message from node j to node i, which is then aggregated as
    # specified in ctor.
    def message(self, x_i, x_j, edge_index):
        # edge_index has shape [2, |E|]
        i_arr, j_arr = edge_index

        # Get the spherical harmonics for each channel
        spherical_harmonic_per_channel = self.getFilterValue(i_arr, j_arr)

        # Calculate the CG Product and multiply each idx by the associated R
        # value.
        cg_products = \
            spherical_harmonic_per_channel.CGproduct(x_j, self.l_filter_)

        return cg_products
    
    def getFilterValue(self, i_arr, j_arr) -> SO3partArr:
        # Get a copy of the spherical harmonics for each channel
        spherical_harmonic_per_channel = \
            self.getSphericalHarmonicsForFilter(i_arr, j_arr, self.in_channels_)
        
        # Apply the learned radial function to distances between i and j arrays.
        distance = self.getDistance(i_arr, j_arr).unsqueeze(-1)
        mlp_vals = torch.stack([mlp(distance) for mlp in self.r_mlps], dim=-1)

        # Combine the two and return it.
        spherical_harmonic_per_channel *= mlp_vals

        return spherical_harmonic_per_channel
        
    # Gets the spherical harmonics assocaited with the self.l_filter for this
    # layer.
    # NOTE: Does NOT depend on channel number. In the original TFN paper, this
    # is the Y_m^{(l_f)}(\hat{r}) spherical harmonic for the filter
    def getSphericalHarmonicsForFilter(
            self, i_arr, j_arr, channel_count) -> SO3part:
        # Get the vector
        i_pos = self.point_positions[i_arr]
        j_pos = self.point_positions[j_arr]
        distance = self.getDistance(i_arr, j_arr)
        assert len(i_pos) == len(j_pos)
        assert len(i_pos) == len(distance)

        vector = (i_pos - j_pos)
        for k in range(len(vector)):
            vector[k] /= distance[k]

        # Extend in a new dimension by copying once per channel
        assert vector.size()[-1] == 3
        vector = vector.unsqueeze(-1).repeat(1, 1, channel_count)

        # Return the Spherical Harmonic associated with it
        return SO3partArr.spharm(self.l_filter, vector)
    
    def getDistance(self, i, j):
        if isinstance(i, int):
            assert isinstance(j, int)
            return self.point_distances_[i,j]
        
        assert len(i) == len(j)
        return torch.tensor(
            [self.point_distances_[i[k],j[k]] for k in range(len(i))])