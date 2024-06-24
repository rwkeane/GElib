import torch

from src.examples.common.point_cloud import PointCloud
from src.examples.common.layers.row_linear import ROWLinear
from src.examples.common.util.radial_bessel_mlp_stack import \
     RadialBesselMlpStack
from src.examples.common.layers.convolution_calculator import \
    ConvolutionCalculator

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
        #
        # TODO: Verify in code that this is in fact rotation-order-wise. It
        # isn't fully clear from the paper.
        self.linear_ = ROWLinear(channels, channels, l_filter, dim = -4)

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
        self.linear_.reset_parameters()

    def forward(self, point_cloud : PointCloud):
        return super().forward(point_cloud)

    # ConvolutionCalculate abstract method.
    def calculateRadialValues(
            self, point_distances : torch.Tensor) -> torch.Tensor:
        # Add an extra dimension so all MLPs can be run in parallel
        distance = point_distances.unsqueeze(-1)

        # Calculate all MLP results.
        mlp_results = self.r_mlps_.forward(distance)
        
        return mlp_results
  
    # ConvolutionCalculator virtual method.
    def getPointCloudRepresentation(self,
                                    point_cloud : PointCloud) -> PointCloud:
        assert point_cloud.dim() >= 5, point_cloud.allSizes()
        assert point_cloud.size()[-4] == self.channels_

        point_cloud = self.linear_.forward(point_cloud)

        assert point_cloud.size()[-4] == self.channels_
        return point_cloud
