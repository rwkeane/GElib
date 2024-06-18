import math
from typing import Any, List, Optional, Sequence, Union
import torch

from gelib import SO3partArr, SO3vecArr

from src.examples.codegen.tensor_recurser_client import TensorRecurserClient
from examples.common.impl.internal_caller import InternalType
from src.examples.common.point_cloud import PointCloud

class PointCloudBase(InternalType, PointCloud, TensorRecurserClient):
    def __init__(self,
                 positions : torch.Tensor,
                 values : SO3vecArr,
                 max_dist : Optional[float] = None):
      self._child_type = None

      super().__init__(child_type = PointCloudBase)

      assert positions != None and isinstance(positions, torch.Tensor), \
          positions
      assert values != None and isinstance(values, SO3vecArr), values

      data_point_count = positions.size()[0]
      if __debug__:
          tau = values.tau()
          for t in tau:
              assert t == data_point_count, values

      self.positions_ = positions
      self.values_ = values

      self.distances_ = torch.cdist(positions, positions)
      assert self.distances_.dim() == 2

      # Creates a complete graph
      edge_list = []
      for i in range(data_point_count):
        for j in range(data_point_count):
          if i == j: 
            continue
          
          if (max_dist == None or 
              math.dist(positions[i], positions[j]) <= max_dist):
            edge_list.append([i, j])

      self.edge_index_ = \
          torch.tensor(edge_list, dtype = torch.int64).t().contiguous()
        
    def edge_list(self) -> torch.Tensor: 
        return self.edge_index_
    
    def positions(self) -> torch.Tensor:
        return self.positions_

    def max_l(self) -> int:
       return self.values_.getLMax()
    
    def getDistance(self,
                    i : Union[torch.tensor, int],
                    j : Union[torch.tensor, int]):
        if isinstance(i, int):
            assert isinstance(j, int)
            return self.distances_[i,j]
        
        assert isinstance(i, torch.Tensor)
        assert isinstance(j, torch.Tensor)
        assert i.size() == j.size()
        if i.dim() == 0:
            return self.getDistance(i.item(), j.item())
        
        return torch.tensor(
            [self.distances_[i[k],j[k]] for k in range(len(i))])
    
    def getPart(self, l_idx : int) -> SO3partArr:
        assert l_idx >= 0 and l_idx <= self.max_l()
        return SO3partArr(self.values_.parts[l_idx])
    
    def CGproduct(self,
                  y : Union[PointCloud, SO3vecArr],
                  max_l : Optional[int] = None) -> PointCloud:
        if isinstance(y, PointCloud):
            y = y.values_
      
        assert isinstance(y, SO3vecArr)
        if max_l == None:
            max_l = -1
        return self.CloneWithNewValue(self.values_.CGproduct(y, max_l))

    def ReducingCGproduct(self,
                          y : Union[PointCloud, SO3vecArr],
                          max_l : Optional[int] = None) -> PointCloud:
        if isinstance(y, PointCloud):
            y = y.values_
        
        assert isinstance(y, SO3vecArr)
        if max_l == None:
            max_l = -1
        return self.CloneWithNewValue(self.values_.ReducingCGproduct(y, max_l))

    def DiagCGproduct(self,
                      y : Union[PointCloud, SO3vecArr],
                      max_l : Optional[int] = None) -> PointCloud:
        if isinstance(y, PointCloud):
            y = y.values_
        
        assert isinstance(y, SO3vecArr)
        if max_l == None:
            max_l = -1
        return self.CloneWithNewValue(self.values_.DiagCGproduct(y, max_l))
    
    def _getVec(self):
       return self.values_
    
    def _createObject(self, vals):
        arr = SO3vecArr()
        arr.parts = vals
        return self.CloneWithNewValue(arr)
    
    @staticmethod
    def CopyAllDataTo(original : 'PointCloudBase',
                      clone : 'PointCloudBase') -> None:
        assert isinstance(clone, PointCloudBase)
        clone.distances_ = original.distances_
        clone.edge_index_ = original.edge_index_
        clone.positions_ = original.positions_
        clone.values_ = original.values_
        InternalType.__init__(clone)
        
        original.addChild(clone)
    
    # Similar to PyTorch functions.
    def allSizes(self, *args, **kwargs):
        return super().size(*args, **kwargs)
    
    def allViews(self, sizes : List, *args, **kwargs):
        assert isinstance(sizes, list), type(sizes)

        parts = self._getParts()
        assert len(sizes) == len(parts)

        results = []
        for i in range(len(sizes)):
            part : torch.Tensor = parts[i]
            kwargs["size"] = sizes[i]
            results.append(part.view(*args, **kwargs))

        return self._createObject(results)

    # Overrides to simplify python usage.
    def size(self, *args, **kwargs) -> torch.Size:
        return self.values_.parts[0].size(*args, **kwargs)
    
    def dim(self, *args, **kwargs) -> int:
        return self.values_.parts[0].dim(*args, **kwargs)
    
    def __len__(self):
        return len(self.values_.parts[0])
    
    def expand_as(self, other):
        assert isinstance(other, PointCloud)
        assert len(self.values_.parts) == len(other.values_.parts)

        results = []
        for i in range(len(self.values_.parts)):
            results.append(self.values_.parts[i].expand_as(
                other.values_.parts[i]))
        
        return self.CloneWithNewValue(SO3partArr(results))
