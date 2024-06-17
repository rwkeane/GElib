from typing import Any, Optional, Sequence, Union
import torch

from gelib import SO3partArr, SO3vecArr
from examples.common.point_cloud import PointCloud
from examples.common.impl.point_cloud_base import PointCloudBase
from examples.common.impl.pyg_point_cloud import PygPointCloud

class PointCloudImpl(PointCloudBase):
    def __init__(self,
                 positions : torch.Tensor,
                 values : SO3vecArr,
                 max_dist : Optional[float] = None):
      super().__init__(positions, values, max_dist)

    @classmethod
    def ClonePointCloudImpl(instance : PointCloudBase,
                            data : Union[SO3partArr, SO3vecArr],
                            l_value : int = -1) -> 'PointCloudImpl':
        # NOTE: Do NOT call super(), because that will be overridded by the
        # other implementation in PygPointCloud, so instead manually copy the
        # code here. Quack Quack....
        clone = PointCloudImpl.__new__(PointCloudImpl)
        clone.distances_ = instance.distances_
        clone.edge_index_ = instance.edge_index_
        clone.positions_ = instance.positions_
        clone.values_ = instance.values_

        return clone

    def ToPygPropegationFormat(self) -> PointCloud:
        return PygPointCloud.Create(PointCloudImpl.ClonePointCloudImpl, self)

    def FromPygPropegationFormat(self) -> PointCloud:
        raise NotImplementedError(
            "This instance is not in PyG Propegation Format!")

    def CloneWithNewValue(self,
                          data : Union[SO3partArr, SO3vecArr],
                          l_value : int = -1) -> 'PointCloudImpl':
        clone = super().CloneWithNewValue(data, l_value)
        assert isinstance(PointCloudImpl)
        return clone