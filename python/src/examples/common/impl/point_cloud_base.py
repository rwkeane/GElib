from abc import abstractmethod
import math
from typing import Any, Iterable, List, Optional, Sequence, Union, overload
import torch

from gelib import SO3partArr, SO3vecArr

from src.examples.codegen.tensor_recurser_client import TensorRecurserClient
from examples.common.impl.util.internal_caller import InternalType
from src.examples.common.point_cloud import PointCloud
import torch.types

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
      assert data_point_count == values.parts[0].size()[-3], values

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

    def max_l(self) -> int:
       return self.values_.getLMax()
    
    def getDistance(self,
                    i : Union[torch.tensor, int],
                    j : Union[torch.tensor, int]) -> torch.Tensor:
        if __debug__:
            if isinstance(i, int):
                assert isinstance(j, int)
            else:
                assert isinstance(i, torch.Tensor)
                assert isinstance(j, torch.Tensor)
                assert i.size() == j.size()

        return torch.tensor(
            [self.distances_[i[k],j[k]] for k in range(len(i))])
    
    def getVectors(self,
                   i : Union[torch.tensor, int],
                   j : Union[torch.tensor, int]) -> torch.Tensor:
        if __debug__:
            if isinstance(i, int):
                assert isinstance(j, int)
            else:
                assert isinstance(i, torch.Tensor)
                assert isinstance(j, torch.Tensor)
                assert i.size() == j.size()

        i_pos = self.positions_[i]
        j_pos = self.positions_[j]
        return i_pos - j_pos
    
    def part(self, l_idx : int) -> SO3partArr:
        assert l_idx >= 0 and l_idx <= self.max_l()
        return SO3partArr(self.values_.parts[l_idx])
    
    def data(self) -> SO3vecArr:
        return self.values_
    
    def CGproduct(self,
                  y : Union[PointCloud, SO3vecArr],
                  max_l : Optional[int] = None) -> PointCloud:
        if isinstance(y, PointCloud):
            y = y.data()
      
        assert isinstance(y, SO3vecArr)
        if max_l == None:
            max_l = -1

        cg_product = self.data().CGproduct(y, max_l)
        return self.CloneWithNewValue(cg_product)

    def ReducingCGproduct(self,
                          y : Union[PointCloud, SO3vecArr],
                          max_l : Optional[int] = None) -> PointCloud:
        if isinstance(y, PointCloud):
            y = y.data()
        
        assert isinstance(y, SO3vecArr)
        if max_l == None:
            max_l = -1
        return self.CloneWithNewValue(self.data().ReducingCGproduct(y, max_l))

    def DiagCGproduct(self,
                      y : Union[PointCloud, SO3vecArr],
                      max_l : Optional[int] = None) -> PointCloud:
        if isinstance(y, PointCloud):
            y = y.data()
        
        assert isinstance(y, SO3vecArr)
        if max_l == None:
            max_l = -1
        return self.CloneWithNewValue(self.data().DiagCGproduct(y, max_l))
    
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

    # PyTorch Geometric support. To be implemented by children.
    @abstractmethod
    def ToPygPropegationFormat(self) -> 'PointCloud':
        """
        Returns a new view of this object which is formatted as expected for
        Pytorch Geometric.
        """
        raise NotImplementedError("This method must be implemented!")

    @abstractmethod
    def FromPygPropegationFormat(self) -> 'PointCloud':
        """
        Converts back from the Pytorch Geometric Specific format to that
        expected by the other parts of the codebase.
        """
        raise NotImplementedError("This method must be implemented!")
    
    # Similar to PyTorch functions.
    def allSizes(self, *args, **kwargs) -> List[torch.Size]:
        return super().size(*args, **kwargs)
    
    def allViews(self, sizes : Iterable, *args, **kwargs) -> 'PointCloudBase':
        assert self.can_access_internals()

        parts = self._getParts()
        assert len(sizes) == len(parts), \
            "{-} vs {1}".format(len(sizes), len(parts))

        results = []
        for i in range(len(sizes)):
            part : torch.Tensor = parts[i]
            kwargs["size"] = sizes[i]
            results.append(part.view(*args, **kwargs))

        return self.CloneWithNewValue(results)

    # Overrides to simplify python usage.
    def size(self, *args, **kwargs) -> torch.Size:
        return self.values_.parts[0].size(*args, **kwargs)
    
    def dim(self, *args, **kwargs) -> int:
        return self.values_.parts[0].dim(*args, **kwargs)
    
    def __len__(self):
        return len(self.values_.parts[0])
    
    def expand_as(self, other):
        return self.expand(other.size())
    
    def expand(self, size: Sequence[Union[int, torch.SymInt]]):
        # TODO: Change this to just a base class call once it better handles
        # equivariance safety.
        assert len(size) == self.dim()
        assert size[-2] == 1

        results = []
        for i in range(len(self.values_.parts)):
            new_size = list(size)
            new_size[-2] == 2 * i + 1
            results.append(self.part(i).expand(tuple(new_size)))
        
        return self.CloneWithNewValue(results)
