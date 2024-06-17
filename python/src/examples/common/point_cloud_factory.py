from typing import Optional
import torch

from gelib import SO3partArr, SO3vecArr

from .point_cloud import PointCloud
from .impl.point_cloud_impl import PointCloudImpl

class PointCloudFactory:
    @staticmethod
    def CreatePointCloud(positions : torch.Tensor,
                         values : SO3vecArr,
                         max_dist : Optional[float] = None) -> PointCloud:
        return PointCloudImpl(positions = positions,
                              values = values,
                              max_dist = max_dist)