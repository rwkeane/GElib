from abc import abstractmethod 
import torch
from torch.nn import Linear, Module, ModuleList

from ...gelib import SO3partArr

class ConvolutionCalculator(Module):
    """
    Performs the actual calculations related to a convolution.

    NOTE: The forward() method has an unusal form. 
    """
    def __init__(self, channels : int, l_filter: int, **kwargs):
        super().__init__(**kwargs)

        assert channels != None
        assert l_filter != None

        self.channels_ : int = channels
        self.l_filter_ : int = l_filter
    
    @abstractmethod
    def calculateRadialValues(
            self, point_distances : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method must be implemented!")

    # Constructs message from node j to node i, which is then aggregated as
    # specified in ctor.
    #
    # TODO: Currently, i->j and j->i are calculated separately even though the
    # value of the CG product is a constant. So this could be updated to be done
    # once instead.
    def forward(self, x_j : SO3partArr, edge_index : torch.Tensor):
        # x_j represents the "source" nodes, and is of shape
        # [<some_num_nodes>, ..., channel_count, 2l_in + 1, N atoms]
        assert isinstance(x_j, SO3partArr)
        assert isinstance(edge_index, torch.Tensor)
        
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
        point_representation = self.getPointRepresentation(x_j)
        assert isinstance(point_representation, torch.Tensor)
        assert point_representation.size() == x_j.size(), \
            "{0} vs {1}".format(point_representation.size(), x_j.size())

        cg_products : SO3partArr = \
            sh_per_channel.DiagCGproduct(point_representation, self.l_filter_)
        
        assert cg_products.size()[-1] == x_j.size()[-1], cg_products.size()
        return cg_products

    def getPointRepresentation(self, x_j : torch.Tensor) -> torch.Tensor:
        # x_j is [<some_num_nodes>, ..., channel_count, 2l_in + 1, N atoms]
        return x_j
    
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
        distance = self.getDistance(i_arr, j_arr)
        radial_values = self.calculateRadialValues(distance)
    
        # Get rid of the extra dimension from MLP application.
        assert radial_values.size()[-2] == 1, radial_values.size()
        radial_values = radial_values.squeeze(-2).transpose(-2, -1)
        assert radial_values.dim() >= 3, radial_values.size()

        return radial_values
        
    # Gets the spherical harmonics assocaited with the self.l_filter for this
    # layer.
    # NOTE: Does NOT depend on channel number. In the original TFN paper, this
    # is the Y_m^{(l_f)}(\hat{r}) spherical harmonic for the filter
    def getSphericalHarmonicsForFilter(self, i_arr, j_arr) -> SO3partArr:
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