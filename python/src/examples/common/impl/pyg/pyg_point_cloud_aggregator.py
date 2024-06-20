from typing import Optional
import torch
from torch_geometric.nn.aggr import Aggregation as PygAggregator

from gelib import SO3vecArr

from src.examples.common.point_cloud import PointCloud
from examples.common.impl.pyg.pyg_point_cloud import PygPointCloud

class PygPointCloudAggregator(PygAggregator):
    """
    A wrapper around the standard PyG aggregator to allow use with a PointCloud.
    """
    def __init__(self, aggregation = "sum"):
        super().__init__()
        self.__agg_type = aggregation

    def forward(self, cloud: PointCloud, index: Optional[torch.Tensor] = None,
            ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            dim: int = -2) -> torch.Tensor:
        assert isinstance(cloud, PygPointCloud)

        results = []
        for i in range(len(cloud.values_.parts)):
            print("Aggregated ", i)
            part = cloud.values_.parts[i]
            size = (cloud.source_size_[i])[0]
            reduction =  self.reduce(
                part, index, ptr, size, dim, reduce=self.__agg_type)
            results.append(reduction)

        return cloud.CloneWithNewValue(SO3vecArr(results))
            