from torch.nn import Linear as TorchLinear

from src.examples.common.point_cloud import PointCloud
from src.examples.common.impl.point_cloud_base import PointCloudBase
from src.examples.common.util.internal_caller import InternalCaller

class Linear(TorchLinear, InternalCaller):
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

    def forward(self, point_cloud : PointCloud) -> PointCloud:
        assert isinstance(point_cloud, PointCloudBase)
        
        # Operate on the correct dimension.
        if self.dim_ != -1:
            point_cloud = point_cloud.transpose(self.dim_, -1)
            
        all_sizes = point_cloud.allSizes()
        point_cloud = point_cloud.allViews(-1, (all_sizes[0])[-1])

        point_cloud = super().forward(point_cloud)
        assert isinstance(point_cloud, PointCloudBase)

        if self.in_features_ != self.out_features_:
          for i in range(len(all_sizes)):
              size = list(all_sizes[i])
              size[-1] == self.out_features_
              all_sizes[i] = tuple(size)

        point_cloud = point_cloud.allViews(all_sizes)

        if self.dim_ != -1:
            point_cloud = point_cloud.transpose(self.dim_, -1)
            
        return point_cloud