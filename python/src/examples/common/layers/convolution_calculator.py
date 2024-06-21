from abc import abstractmethod
from enum import Enum 
import torch
from torch.nn import Linear, Module, ModuleList

from gelib import SO3partArr, SO3vecArr

from src.examples.common.point_cloud import PointCloud
from examples.common.impl.util.internal_caller import InternalCaller

class ConvolutionCalculator(InternalCaller, Module):
    """
    Performs the actual calculations related to a convolution.

    NOTE: All inputs are assumed to have the same l value for a given layer. 
    Changing this would mainly require updating calculateRadialValues() to be
    per l_i value.
    """
    def __init__(self,
                 channels : int,
                 l_filter: int,
                 l_max : int,
                 **kwargs):
        super().__init__(**kwargs)

        assert channels != None
        assert l_filter != None

        self.channels_ : int = channels
        self.l_filter_ : int = l_filter
        self.l_max_ = l_max
    
    @abstractmethod
    def calculateRadialValues(
            self, point_distances : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method must be implemented!")
    
    def reset_parameters(self):
        pass

    # Constructs message from node j to node i, which is then aggregated as
    # specified in ctor.
    #
    # TODO: Currently, i->j and j->i are calculated separately even though the
    # value of the CG product is a constant. So this could be updated to be done
    # once instead.
    def forward(self, point_cloud : PointCloud) -> PointCloud:
        # x_j represents the "source" nodes, and is of shape
        # [<some_num_nodes>, ..., channel_count, 2l_in + 1, N atoms]
        assert isinstance(point_cloud, PointCloud)
        
        # edge_index has shape [2, |E|].
        assert point_cloud.edge_list().size()[0] == 2
        i_arr = point_cloud.edge_list()[0,:]
        j_arr = point_cloud.edge_list()[1,:]

        # Get the spherical harmonics for each channel.
        sh_per_channel = self.getFilterValue(i_arr, j_arr, point_cloud)
        assert isinstance(sh_per_channel, SO3partArr)
        
        # Add dimensions to spherical harmonics to match x_j.
        assert sh_per_channel.dim() <= point_cloud.dim(), \
            "{0} vs {1}".format(sh_per_channel.dim(), point_cloud.dim())
        while sh_per_channel.dim() < point_cloud.dim():
            sh_per_channel = sh_per_channel.unsqueeze(0)
        new_size = list(sh_per_channel.size())
        new_size[-2] = sh_per_channel.size()[-2]
        sh_per_channel = sh_per_channel.expand(tuple(new_size))
        sh_per_channel = \
            point_cloud.CloneWithNewValue(sh_per_channel, self.l_max_)
            
        # Calculate CG product.
        representation = self.getPointCloudRepresentation(point_cloud)
        print("representation", representation)

        return representation










        assert isinstance(representation, PointCloud)
        assert representation.size()[:-2] == point_cloud.size()[:-2], \
            "{0} vs {1}".format(representation.size(), point_cloud.size())

        # sh_per_channel.expand_as(representation)
        cg_products = representation.CGproduct(sh_per_channel, self.l_max_)
        
        return cg_products

    def getPointCloudRepresentation(self, x_j : PointCloud) -> PointCloud:
        # x_j is [<some_num_nodes>, ..., channel_count, 2 * l_in + 1, N atoms]
        return x_j
    
    def getFilterValue(self,
                       i_arr : torch.Tensor,
                       j_arr : torch.Tensor,
                       point_cloud : PointCloud) -> SO3partArr:
        # Get a copy of the spherical harmonics for each channel
        sh_per_channel = self.getSphericalHarmonicsForFilter(
            i_arr, j_arr, point_cloud)
        assert isinstance(sh_per_channel, SO3partArr)
        
        # Apply the learned radial function to distances between i and j arrays.
        mlp_vals = self.getRValues(i_arr, j_arr, point_cloud)
        assert isinstance(mlp_vals, torch.Tensor), type(mlp_vals)
        assert mlp_vals.dim() == sh_per_channel.dim() and \
               mlp_vals.size()[0:-4] == sh_per_channel.size()[0:-4] and \
               mlp_vals.size()[-2:-1] == sh_per_channel.size()[-2:-1], \
            "{0} vs {1}".format(mlp_vals.size(), sh_per_channel.size())
        sh_per_channel = sh_per_channel.expand_as(mlp_vals)

        # Combine the two and return it.
        sh_per_channel = sh_per_channel * mlp_vals
        assert sh_per_channel.dim() == 3, sh_per_channel.size()
        
        return sh_per_channel
    
    def getRValues(self,
                   i_arr : torch.Tensor,
                   j_arr : torch.Tensor,
                   point_cloud : PointCloud) -> torch.Tensor:
        distance = point_cloud.getDistance(i_arr, j_arr)
        radial_values = self.calculateRadialValues(distance)
    
        # Get rid of the extra dimension from MLP application.
        assert radial_values.size()[-2] == 1, radial_values.size()
        radial_values = radial_values.squeeze(-2).permute(-2, -1, -3)
        assert radial_values.dim() >= 3, radial_values.size()

        return radial_values
        
    # Gets the spherical harmonics assocaited with the self.l_filter for this
    # layer.
    # NOTE: Does NOT depend on channel number. In the original TFN paper, this
    # is the Y_m^{(l_f)}(\hat{r}) spherical harmonic for the filter
    def getSphericalHarmonicsForFilter(
            self,
            i_arr : torch.Tensor,
            j_arr : torch.Tensor,
            point_cloud : PointCloud) -> SO3partArr:
        # Get the vector
        distance = point_cloud.getDistance(i_arr, j_arr)
        vector = point_cloud.getVectors(i_arr, j_arr)
        for k in range(len(vector)):
            vector[k] /= distance[k]

        # Extend in a new dimension by copying once per channel
        # spharm expects (batch_size, 3 (dimensions), N)
        assert vector.size()[-1] == 3
        vector = vector.unsqueeze(-1)

        # Return the Spherical Harmonic associated with it.
        return SO3partArr.spharm(self.l_filter_, vector)