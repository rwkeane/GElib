from abc import abstractmethod
import math
from typing import Any, Optional, Sequence, Union

import torch
from torch_geometric.data import Data as PygData
from torch_geometric.nn import MessagePassing

from gelib import SO3partArr, SO3vecArr

from src.examples.codegen.tensor_recurser import TensorRecurser

class PointCloud(TensorRecurser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @abstractmethod
    def edge_list(self) -> torch.Tensor: 
        raise NotImplementedError("This method must be implemented!")
    
    @abstractmethod
    def positions(self) -> torch.Tensor:
        raise NotImplementedError("This method must be implemented!")

    @abstractmethod
    def maxL(self) -> int:
        raise NotImplementedError("This method must be implemented!")
    
    @abstractmethod
    def getDistance(self,
                    i : Union[torch.tensor, int],
                    j : Union[torch.tensor, int]):
        raise NotImplementedError("This method must be implemented!")
    
    @abstractmethod
    def asGraphData(self, l_value : int) -> PygData:
        raise NotImplementedError("This method must be implemented!")
    
    @abstractmethod
    def CGproduct(self,
                  y : Union['PointCloud', SO3vecArr],
                  maxl : Optional[int] = None) -> 'PointCloud':
        raise NotImplementedError("This method must be implemented!")

    @abstractmethod
    def ReducingCGproduct(self,
                          y : Union['PointCloud', SO3vecArr],
                          maxl : Optional[int] = None) -> 'PointCloud':
        raise NotImplementedError("This method must be implemented!")

    @abstractmethod
    def DiagCGproduct(self,
                      y : Union['PointCloud', SO3vecArr],
                      maxl : Optional[int] = None) -> 'PointCloud':
        raise NotImplementedError("This method must be implemented!")

    @abstractmethod
    def CloneWithNewValue(self,
                          data : Union[PygData, SO3partArr, SO3vecArr],
                          l_value : int = -1) -> 'PointCloud':
        raise NotImplementedError("This method must be implemented!")

    @abstractmethod
    def ToPygPropegationFormat(self) -> 'PointCloud':
        raise NotImplementedError("This method must be implemented!")

    @abstractmethod
    def FromPygPropegationFormat(self) -> 'PointCloud':
        raise NotImplementedError("This method must be implemented!")

    
  