import math
from typing import Any, Optional, Sequence, Union
import torch

from gelib import SO3partArr, SO3vecArr

from examples.codegen.tensor_recurser_client import TensorRecurserClient
from examples.common.point_cloud import PointCloud

class PointCloudBase(TensorRecurserClient, PointCloud):
    def __init__(self,
                 positions : torch.Tensor,
                 values : SO3vecArr,
                 max_dist : Optional[float] = None):
      super().__init__(child_type = PointCloudBase)

      assert positions != None and isinstance(positions, torch.Tensor), \
          positions
      assert values != None and isinstance(values, SO3vecArr), values

      data_point_count = positions.size()[0]
      assert values.getN() == data_point_count, values

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

    def maxL(self) -> int:
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
    
    def CGproduct(self,
                  y : Union[PointCloud, SO3vecArr],
                  maxl : Optional[int] = None) -> PointCloud:
        if isinstance(y, PointCloud):
            y = y.values()
      
        assert isinstance(y, SO3vecArr)
        if maxl == None:
            maxl = -1
        return self.CloneWithNewValue(self.values_.CGproduct(y, maxl))

    def ReducingCGproduct(self,
                          y : Union[PointCloud, SO3vecArr],
                          maxl : Optional[int] = None) -> PointCloud:
        if isinstance(y, PointCloud):
            y = y.values()
        
        assert isinstance(y, SO3vecArr)
        if maxl == None:
            maxl = -1
        return self.CloneWithNewValue(self.values_.ReducingCGproduct(y, maxl))

    def DiagCGproduct(self,
                      y : Union[PointCloud, SO3vecArr],
                      maxl : Optional[int] = None) -> PointCloud:
        if isinstance(y, PointCloud):
            y = y.values()
        
        assert isinstance(y, SO3vecArr)
        if maxl == None:
            maxl = -1
        return self.CloneWithNewValue(self.values_.DiagCGproduct(y, maxl))

    def CloneWithNewValue(self,
                          data : Union[SO3partArr, SO3vecArr],
                          l_value : int = -1) -> PointCloud:
        assert l_value >= 0 or isinstance(data, SO3vecArr)
        
        if isinstance(data, SO3partArr):
           return self.CloneWithNewValue(data.asVec(), l_value)
        
        assert isinstance(data, SO3vecArr)

        # Clone as either a PointCloudImpl or a type that descends from it.
        clone = self.type.__new__(self.type)
        assert isinstance(clone, PointCloudBase)

        # Copy all data
        clone.distances_ = self.distances_
        clone.edge_index_ = self.edge_index_
        clone.positions_ = self.positions_
        clone.values_ = data

        return clone
    
    def __getParts(self):
       return self.values_.parts
    
    def __createObject(self, vals):
        arr = SO3vecArr()
        arr.parts = vals
        return self.CloneWithNewValue(arr)

    # Overrides to simplify python usage.
    def size(self, *args, **kargs) -> torch.Size:
        size = super().size(*args, **kargs)
        return size[0]
    
    def dim(self, *args, **kargs) -> int:
        dim = super().dim(*args, **kargs)
        return dim[0]
    
    def __len__(self):
        return len(self.values_.parts[0])