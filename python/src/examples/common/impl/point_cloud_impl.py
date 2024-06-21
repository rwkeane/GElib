from typing import Any, Iterable, List, Optional, Sequence, Union
import torch

from gelib import SO3partArr, SO3vecArr

from src.examples.common.point_cloud import PointCloud
from src.examples.common.impl.point_cloud_base import PointCloudBase
from examples.common.impl.pyg.pyg_point_cloud import PygPointCloud

class PointCloudImpl(PointCloudBase):
    def __init__(self,
                 positions : torch.Tensor,
                 values : SO3vecArr,
                 max_dist : Optional[float] = None):
      super().__init__(positions, values, max_dist)

    @staticmethod
    def ClonePointCloudImpl(instance : PointCloudBase) -> 'PointCloudImpl':
        clone = PointCloudImpl.__new__(PointCloudImpl)
        PointCloudBase.CopyAllDataTo(instance, clone)

        return clone

    def ToPygPropegationFormat(self) -> PointCloud:
        return PygPointCloud.Create(self, PointCloudImpl.ClonePointCloudImpl)

    def FromPygPropegationFormat(self) -> PointCloud:
        raise NotImplementedError(
            "This instance is not in PyG Propegation Format!")

    def CloneWithNewValue(
            self,
            data : Union[SO3partArr, SO3vecArr, List[SO3partArr]],
            l_value : int = -1) -> PointCloud:
        assert l_value >= 0 or isinstance(data, SO3vecArr)
        
        if isinstance(data, SO3partArr):
           data = SO3vecArr.from_part(data, l_value)

        elif not isinstance(data, SO3vecArr) and hasattr(data, '__iter__'):
            assert len(data) > 0
            if isinstance(data[0], torch.Tensor):
                vec = SO3vecArr()
                vec.parts = list(data)
                data = vec
        
        assert isinstance(data, SO3vecArr)

        # Clone as either a PointCloudImpl or a type that descends from it.
        clone = PointCloudImpl.__new__(PointCloudImpl)
        assert isinstance(clone, PointCloudImpl)

        # Copy all data
        PointCloudBase.CopyAllDataTo(self, clone)
        clone.values_ = data

        return clone