from torch.nn import Linear as TorchLinear

from src.examples.common.point_cloud import PointCloud
from src.examples.common.impl.point_cloud_base import PointCloudBase
from examples.common.impl.util.internal_caller import InternalCaller

class Linear(InternalCaller, TorchLinear):
    """
    Wrapper around PyTorch Linear layer to avoid exposing dimension-swapping,
    which could break equivariance.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dim : int = -1,
                 bias: bool = True,
                 device = None) -> None:
        super().__init__(in_features, out_features, bias, device)

        self.in_features_ = in_features
        self.out_features_ = out_features
        self.dim_ = dim
        
        assert self.dim_ != -2, \
            "Modifying dimension -2 would break equivariance!"

    def forward(self, point_cloud : PointCloud) -> PointCloud:
        assert isinstance(point_cloud, PointCloudBase)
        assert self.dim_ - point_cloud.dim() != -2, \
            "Modifying dimension -2 would break equivariance!"
        
        # Operate on the correct dimension.
        if self.dim_ != -1:
            point_cloud = point_cloud.transpose(self.dim_, -1)

        point_cloud = super().forward(point_cloud)
        assert isinstance(point_cloud, PointCloudBase)

        if self.dim_ != -1:
            point_cloud = point_cloud.transpose(self.dim_, -1)
            
        return point_cloud