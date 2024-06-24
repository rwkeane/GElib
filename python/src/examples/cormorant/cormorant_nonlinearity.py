import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential

from gelib import SO3partArr, SO3vecArr

from src.examples.common.util.message_passing import MessagePassing

class CormorantNonlinearity(MessagePassing):
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 l_filter: int,
                 point_positions: torch.tensor):
        super().__init__(aggr='add')  # "Add" aggregation
        self.lin = Linear(in_channels, out_channels, bias=False)

        self.in_channels : int = in_channels
        self.out_channels : int = out_channels
        self.l_filter : int = l_filter
        self.point_positions = point_positions
        self.point_distances = torch.cdist(point_positions, point_positions)

        # Create a list to store MLPs for each output channel to handle learning
        # the radial function for each channel, as a function of the distance
        # between 2 points.
        # TODO: Update this value
        kNumHiddenLayerNodes = 64
        self.r_mlps = torch.nn.ModuleList([
            Sequential(
                Linear(1, kNumHiddenLayerNodes), 
                ReLU(),
                Linear(kNumHiddenLayerNodes, 1)  # Output is a single value R_c
            ) for _ in range(self.in_channels)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        for r_mlp in self.r_mlps:
            for layer in r_mlp:
                if isinstance(layer, Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x, edge_index, edge_attr):
        # TODO: This may or may not be needed.
        # x = self.lin(x)

        # Start propagating messages.
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    # Constructs message from node j to node i, which is then aggregated as
    # specified in ctor.
    def message(self, x_i, x_j, edge_index):
        i, j = edge_index

        # Get he spherical harmonics for each channel
        spherical_harmonic_arr = self.getFilterValue(i, j)

        # Calculate the CG Product and multiply each idx by the associated R
        # value.
        cg_products = spherical_harmonic_arr.CGproduct(x_j)

        return cg_products
    
    def getFilterValue(self, i, j) -> SO3partArr:
        # Get a copy of the spherical harmonics for each channel
        spherical_harmonic = self.getSphericalHarmonicsForFilter(i, j)
        spherical_harmonic_arr = \
            SO3partArr.createCopies(spherical_harmonic, self.in_channels)
        assert len(spherical_harmonic_arr) == len(self.r_mlps)
        
        # Apply the learned radial function
        distance = self.point_distances[i,j]
        for i in range(self.in_channels):
            spherical_harmonic_arr[i] *= self.r_mlps[i](distance)

        return spherical_harmonic_arr
        
    # Gets the spherical harmonics assocaited with the self.l_filter for this
    # layer.
    # NOTE: Does NOT depend on channel number. In the original TFN paper, this
    # is the Y_m^{(l_f)}(\hat{r}) spherical harmonic for the filter
    def getSphericalHarmonicsForFilter(self, i, j) -> SO3partArr:
        # Get the vector
        i_pos = self.point_positions[i]
        j_pos = self.point_positions[j]
        vector = (i_pos - j_pos) / self.point_distances[i,j]

        # Return the Spherical Harmonic associated with it
        return SO3partArr.spharm(self.l_filter, vector)