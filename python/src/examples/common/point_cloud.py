from abc import abstractmethod
import math
from typing import  List, Optional, Sequence, Union

import torch
from torch_geometric.data import Data as PygData

from gelib import SO3partArr, SO3vecArr

from src.examples.codegen.tensor_recurser import TensorRecurser

class PointCloud(TensorRecurser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    # Accessing data related to the underlying graph and tensor representations.
    @abstractmethod
    def edge_list(self) -> torch.Tensor: 
        """
        Returns the edge list for the underlying graph.
        """
        raise NotImplementedError("This method must be implemented!")

    @abstractmethod
    def max_l(self) -> int:
        """
        Returns the maximum L value of the underlying SO3vecArr.
        """
        raise NotImplementedError("This method must be implemented!")
    
    def positions(self) -> torch.Tensor:
        """
        Returns the underlying point positions.

        TODO: Remove this function to ensure equivariance is maintained.
        """
        raise NotImplementedError("This method must be implemented!")
    
    @abstractmethod
    def getDistance(self,
                    i : Union[torch.tensor, int],
                    j : Union[torch.tensor, int]):
        """
        Returns the distance between |i| and |j|.
        """
        raise NotImplementedError("This method must be implemented!")
    
    @abstractmethod
    def getPart(self, l_idx : int) -> SO3partArr:
        """
        Returns the part associated with a specific l value |l_idx|.
        """
        raise NotImplementedError("This method must be implemented!")

    
    # CG product calculations.
    @abstractmethod
    def CGproduct(self,
                  y : Union['PointCloud', SO3vecArr],
                  max_l : Optional[int] = None) -> 'PointCloud':
        """
        Calculates the Clebsch-Gordan product of this instance with another
        instance.
        """
        raise NotImplementedError("This method must be implemented!")

    @abstractmethod
    def ReducingCGproduct(self,
                          y : Union['PointCloud', SO3vecArr],
                          max_l : Optional[int] = None) -> 'PointCloud':
        """
        Calculates the Reducing Clebsch-Gordan product of this instance with
        another instance.
        """
        raise NotImplementedError("This method must be implemented!")

    @abstractmethod
    def DiagCGproduct(self,
                      y : Union['PointCloud', SO3vecArr],
                      max_l : Optional[int] = None) -> 'PointCloud':
        """
        Calculates the Diagonal Clebsch-Gordan product of this instance with
        another instance.
        """
        raise NotImplementedError("This method must be implemented!")

    # Wrappers used for supporting PyTorch-like functionality.
    @abstractmethod
    def CloneWithNewValue(self,
                          data : Union[PygData, SO3partArr, SO3vecArr],
                          l_value : int = -1) -> 'PointCloud':
        """
        Creates a clone of this oject with the new value |data|. If |data| is
        an SO3partArr, it is converted to an SO3vecArr with maximum l value 
        |l_value|.
        """
        raise NotImplementedError("This method must be implemented!")

    # PyTorch Geometric support.
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

    # PyTorch-like methods to be used in special cases instead of the standard
    # PyTorch ones.
    @abstractmethod
    def allSizes(self, *args, **kwargs) -> List[torch.Size]:
        """
        Returns a list containing the result of torch.Tensor.size() when called
        on each part of the underlying SO3vecArr. Any additional parameters are
        passed to all size() calls.
        """
        raise NotImplementedError("This method must be implemented!")
    
    @abstractmethod
    def allViews(self, sizes : List[Sequence[int | torch.SymInt]],
                 *args,
                 **kwargs) -> 'PointCloud':
        """
        Returns a PointCloud with entries representing where torch.Tensor.view()
        is called on the underlying part i with the size stored in |sizes[i]|.
        Any additional parameters are passed to all view() calls.
        """
        raise NotImplementedError("This method must be implemented!")