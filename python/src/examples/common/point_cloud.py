from ctypes import Union
import math
from typing import Any, Optional, Sequence

import torch
from torch_geometric.data import Data as PygData
from torch_geometric.nn import MessagePassing

from ...gelib import SO3partArr, SO3vecArr

from src.examples.common.pyg_helper import \
    reshapeInputForPyg, undoReshapeInputForPyg
from src.examples.codegen.tensor_recurser import TensorRecurser

class PointCloud(TensorRecurser):
    def __init__(self,
                 positions : torch.Tensor,
                 values : SO3vecArr,
                 max_dist : Optional[float] = None):
      super().__init__()

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
    
    def asGraphData(self, l_value : int) -> PygData:
        assert l_value >= 0
        assert l_value <= self.values_.getLMax()
        
        data = PygData(x = self.values_.getPart(l_value),
                      edge_index = self.edge_index_)
        data.x = reshapeInputForPyg(data.x)
        data.validate(raise_on_error = True)
        return data
    
    def CGproduct(self,
                  y : Union['PointCloud', SO3vecArr],
                  maxl : Optional[int] = None) -> 'PointCloud':
        if isinstance(y, PointCloud):
            y = y.values_
      
        assert isinstance(y, SO3vecArr)
        if maxl == None:
            maxl = -1
        return self.CloneWithNewValue(self.values_.CGproduct(y, maxl))

    def ReducingCGproduct(self,
                          y : Union['PointCloud', SO3vecArr],
                          maxl : Optional[int] = None) -> 'PointCloud':
        if isinstance(y, PointCloud):
            y = y.values_
        
        assert isinstance(y, SO3vecArr)
        if maxl == None:
            maxl = -1
        return self.CloneWithNewValue(self.values_.ReducingCGproduct(y, maxl))

    def DiagCGproduct(self,
                      y : Union['PointCloud', SO3vecArr],
                      maxl : Optional[int] = None) -> 'PointCloud':
        if isinstance(y, PointCloud):
            y = y.values_
        
        assert isinstance(y, SO3vecArr)
        if maxl == None:
            maxl = -1
        return self.CloneWithNewValue(self.values_.DiagCGproduct(y, maxl))

    @staticmethod
    def CloneWithNewValue(self,
                          data : Union[PygData, SO3partArr, SO3vecArr],
                          l_value : int = -1) -> 'PointCloud':
        assert l_value >= 0 or isinstance(data, SO3vecArr)

        if isinstance(data, PygData):
          x = undoReshapeInputForPyg(data.x)
          return PointCloud.CloneWithNewValue(x, l_value)
        
        if isinstance(data, SO3partArr):
           return PointCloud.CloneWithNewValue(data.asVec(), l_value)
        
        assert isinstance(data, SO3vecArr)
        clone = PointCloud.__new__(PointCloud)
        clone.distances_ = self.distances_
        clone.edge_index_ = self.edge_index_
        clone.positions_ = self.positions_
        clone.values_ = data

        return clone
    
    def __getParts(self):
       return self.values_.parts
    
    def __createObject(self, vals):
        clone = PointCloud.__new__(PointCloud)
        clone.distances_ = self.distances_
        clone.edge_index_ = self.edge_index_
        clone.positions_ = self.positions_
        clone.values_ = SO3vecArr()
        clone.values_.parts = vals

        return clone

    # Overrides to simplify python usage.
    def size(self, *args, **kargs) -> torch.Size:
        size = super().size(*args, **kargs)
        return size[0]
    
    def dim(self, *args, **kargs) -> int:
        dim = super().dim(*args, **kargs)
        return dim[0]
    
    def expand_as(self,
                  other : Union['PointCloud', torch.Tensor],
                  *args,
                  **kwargs) -> 'PointCloud':
        if isinstance(other, PointCloud):
            assert other.maxL() == self.maxL()

            results = []
            for i in range(self.maxL() + 1):
                results.append(self.values_.parts[i].expand_as(
                    other.values_.parts[i]), *args, **kwargs)
            return self.CloneWithNewValue(SO3vecArr(results))
        
        return self.expand(other.size(), *args, **kwargs)
    
    def expand(self,
               size : Sequence[Union[int, torch.SymInt]],
               *args,
               **kwargs) -> 'PointCloud':
        results = []
        size = list(size)
        for i in range(self.maxL() + 1):
            size[-2] = self.values_.parts[i].size()[-2]
            results.append(
                self.values_.parts[i].expand(tuple(size)), *args, **kwargs)
            
        return self.CloneWithNewValue(SO3vecArr(results))
    


    # TODO: Add code generation for this special case? Can use suggestion from the AI.
    
    def __len__(self):
        return len(self.values_.parts[0])
    
    def __add__(self, other : Any):
        if isinstance(other, torch.Tensor):
            return None
        return super().__add__(other)
    
    def __sub__(self, other : Any):
        pass
    
    def __mul__(self, other : Any):
        pass
    
    def __pow__(self, other : Any):
        pass
    
    def __truediv__(self, other : Any):
        pass
    
    def __floordiv__(self, other : Any):
        pass
    
    def __mod__(self, other : Any):
        pass
    
    def __and__(self, other : Any):
        pass
    
    def __or__(self, other : Any):
        pass
    
    def __xor__(self, other : Any):
        pass
    
    def __lt__(self, other : Any):
        pass
    
    def __le__(self, other : Any):
        pass
    
    def __eq__(self, other : Any):
        pass
    
    def __ne__(self, other : Any):
        pass
    
    def __ne__(self, other : Any):
        pass
    
    def __ne__(self, other : Any):
        pass
    
    def __ne__(self, other : Any):
        pass
Tensor.__len__
Tensor.__add__
Tensor.__call__
Tensor.__sub__
Tensor.__mul__
Tensor.__pow__
Tensor.__truediv__
Tensor.__floordiv__
Tensor.__mod__
Tensor.__lshift__
Tensor.__rshift__
Tensor.__and__
Tensor.__or__
Tensor.__xor__
Tensor.__invert__
Tensor.__lt__
Tensor.__le__
Tensor.__eq__
Tensor.__ne__
Tensor.__gt__
Tensor.__ge__