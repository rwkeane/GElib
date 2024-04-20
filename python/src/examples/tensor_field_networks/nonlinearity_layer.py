
import math
import torch
from torch_geometric.nn import MessagePassing
from typing import Any, Callable, Generic, List, TypeVar

from ...gelib import SO3partArr

class TfnNonlinearityLayer(MessagePassing):
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 nonlinearity_fn):
        super().__init__(aggr='add')  # "Add" aggregation

        self.in_channels : int = in_channels 
        self.out_channels : int = out_channels
        self.nonlinearity = nonlinearity_fn

        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # TODO: This may or may not be needed.
        # x = self.lin(x)

        # Calculate R_c values using MLPs
        r_values_list = []
        for r_mlp in self.r_mlps:
            r_values = r_mlp(self.point_distances)
            r_values_list.append(r_values)
        r_values = torch.cat(r_values_list, dim=-1)  # Combine outputs

        # Apply R_c values to features
        x = x * r_values.unsqueeze(-1)

        # Start propagating messages.
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    # Constructs message from node j to node i, which is then aggregated as
    # specified in ctor.
    def message(self, x_i, x_j, edge_index):
        i, j = edge_index

        # Get a copy of the spherical harmonics for each channel
        spherical_harmonic = self.getSphericalHarmonicsForMessage(i, j)
        spherical_harmonic_arr = \
            SO3partArr.createCopies(spherical_harmonic, self.in_channels)
        
        # Calculate the CG Product and multiply each idx by the associated R
        # value.
        cg_products = spherical_harmonic_arr.CGproduct(x_j)

        return cg_products
        