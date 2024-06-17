from typing import Callable, Union

from gelib import SO3partArr, SO3vecArr

from src.examples.common.impl.point_cloud_base import PointCloudBase
from src.examples.common.point_cloud import PointCloud
from src.examples.common.impl.pyg_helper_impl import \
    flattenForPygPropegate, undoFlattenForPygPropegate, reshapeInputForPyg, \
        undoReshapeInputForPyg

class PygPointCloud(PointCloudBase):
    def __init__(self):
        # Define attributes just for the code hints.
        self.original_size_ = None
        self.original_clone_func_ = None

        raise NotImplementedError("Must be created with __new__()!")
    
    @staticmethod
    def Create(original : PointCloudBase,
               clone_func : Callable[[PointCloudBase], PointCloudBase]) \
                  -> 'PygPointCloud':
        assert isinstance(original, PointCloudBase), type(original)

        copy = reshapeInputForPyg(original)
        copy, size = flattenForPygPropegate(copy)
        assert isinstance(copy, PointCloudBase), type(copy)

        clone = PygPointCloud.__new__(PygPointCloud)
        clone.distances_ = copy.distances_
        clone.edge_index_ = copy.edge_index_
        clone.positions_ = copy.positions_
        clone.values_ = copy.values_
        clone._child_type = copy._child_type
        clone.original_size_ = size
        clone.original_clone_func_ = clone_func

        return clone

    def ToPygPropegationFormat(self) -> PointCloud:
        raise NotImplementedError("This instance is already in PyG Format!")

    def FromPygPropegationFormat(self) -> PointCloud:
        assert self.original_size_ != None
        copy = undoFlattenForPygPropegate(self, self.original_size_)
        copy = undoReshapeInputForPyg(copy)
        copy = self.original_clone_func_(copy)
        assert not isinstance(copy, PygPointCloud)
        return copy

    def CloneWithNewValue(self,
                          data : Union[SO3partArr, SO3vecArr],
                          l_value : int = -1) -> PointCloud:
        assert l_value >= 0 or isinstance(data, SO3vecArr)
        
        if isinstance(data, SO3partArr):
           data = data.asVec(l_value)
        
        assert isinstance(data, SO3vecArr)

        # Clone as either a PointCloudImpl or a type that descends from it.
        clone = PygPointCloud.__new__(PygPointCloud)
        assert isinstance(clone, PygPointCloud)

        # Copy all data
        clone.distances_ = self.distances_
        clone.edge_index_ = self.edge_index_
        clone.positions_ = self.positions_
        clone.values_ = data
        clone.original_size_ = self.original_size_
        clone.original_clone_func_ = self.original_clone_func_
        clone._child_type = self._child_type

        return clone