from abc import abstractmethod
from typing import  Iterable, List, Optional, Sequence, Union
import uuid

import torch

from gelib import SO3partArr, SO3vecArr

from src.examples.codegen.tensor_recurser import TensorRecurser

class PointCloud(TensorRecurser):
    """
    Represents a point cloud, and acts as a wrapper around SO3vecArr while
    exposing a number of useful operations and enforcing that equivariance is
    maintained when such operations are used.

    New instances should be constructed using PointCloudFactory, or by using
    methods of this class.
    """
    def __init__(self, *args, **kwargs):
        self.id_ = uuid.uuid4()
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
    
    @abstractmethod
    def data(self) -> SO3vecArr:
        """
        Returns the SO3vecArr backing this instance.
        """
        raise NotImplementedError("This method must be implemented!")
    
    @abstractmethod
    def part(self, l_idx : int) -> SO3partArr:
        """
        Returns the part associated with a specific l value |l_idx|.
        """
        raise NotImplementedError("This method must be implemented!")
    
    @abstractmethod
    def getDistance(self,
                    i : Union[torch.tensor, int],
                    j : Union[torch.tensor, int]) -> torch.Tensor:
        """
        Returns the distance between |i| and |j|.
        """
        raise NotImplementedError("This method must be implemented!")
    
    @abstractmethod
    def getVectors(self,
                   i : Union[torch.tensor, int],
                   j : Union[torch.tensor, int]) -> torch.Tensor:
        """
        Returns the vector pointing from i to j.
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
    def CloneWithNewValue(
            self,
            data : Union[SO3partArr, SO3vecArr, Iterable[SO3partArr]],
            l_value : int = -1) -> 'PointCloud':
        """
        Creates a clone of this oject with the new value |data|. If |data| is
        an SO3partArr, it is converted to an SO3vecArr with maximum l value 
        |l_value|.
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
    
    def Clone(self) -> 'PointCloud':
        return self.CloneWithNewValue(self.data())
    
    def __str__(self):
        # return self.data().__str__()
        return "ID: {0} type: {1}".format(self.id_, type(self))