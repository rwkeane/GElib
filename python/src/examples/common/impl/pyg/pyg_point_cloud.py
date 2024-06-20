from typing import Callable, Union

from gelib import SO3partArr, SO3vecArr

from src.examples.common.impl.point_cloud_base import PointCloudBase
from src.examples.common.point_cloud import PointCloud
from examples.common.impl.pyg.pyg_helper_impl import \
    flattenForPygPropegate, undoFlattenForPygPropegate, reshapeInputForPyg, \
        undoReshapeInputForPyg

class PygPointCloud(PointCloudBase):
    def __init__(self):
        # Define attributes just for the code hints.
        self.source_size_ = None
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
        PointCloudBase.CopyAllDataTo(copy, clone)
        clone.original_clone_func_ = clone_func
        clone.source_size_ = None

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
        PointCloudBase.CopyAllDataTo(self, clone)
        clone.values_ = data
        clone.original_clone_func_ = self.original_clone_func_
        clone.source_size_ = self.source_size_

        return clone