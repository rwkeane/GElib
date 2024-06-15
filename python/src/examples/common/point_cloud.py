from ctypes import Union
import math
from typing import Optional

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
                 values : Union[SO3vecArr, SO3partArr],
                 max_dist : float = None,
                 max_l : int = None):
      super().__init__()

      if isinstance(values, SO3partArr):
         assert max_l != None and isinstance(max_l, int)
         values = values.asVec(max_l)

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
    
    def __getitem__(self, l_value) -> SO3partArr: 
        assert isinstance(l_value, int)

        result = self.values_.parts[l_value]

        if not isinstance(result, SO3partArr):
           assert isinstance(result, torch.Tensor)
           result = SO3partArr(result)

        return result

    def __setitem__(self, l_value, data): 
        assert isinstance(l_value, int)

        if isinstance(data, PygData):
            data = undoReshapeInputForPyg(data.x)
        
        assert isinstance(data, SO3partArr)
        self.values_.parts[l_value] = data
    
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
    
    def CGproduct(
          self, y : 'PointCloud', maxl : Optional[int] = -1) -> 'PointCloud':
       return self.CloneWithNewValue(self.values_.CGproduct(y.CGproduct, maxl))

    def ReducingCGproduct(
          self, y : 'PointCloud', maxl : Optional[int] =-1) -> 'PointCloud':
       return self.CloneWithNewValue(
            self.values_.ReducingCGproduct(y.CGproduct, maxl))

    def DiagCGproduct(
          self, y : 'SO3vecArr', maxl : Optional[int] =-1) -> 'SO3vecArr':
       return self.CloneWithNewValue(
            self.values_.DiagCGproduct(y.CGproduct, maxl))
    
    def getParts(self):
       return self.values_.parts
    
    def createObject(self, vals):
        clone = PointCloud.__new__(PointCloud)
        clone.distances_ = self.distances_
        clone.edge_index_ = self.edge_index_
        clone.positions_ = self.positions_
        clone.values_ = SO3vecArr()
        clone.values_.parts = vals

        return clone

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