from abc import abstractmethod, property
from typing import List

from gelib import Point, WeightRegistry

class Point(WeightRegistry):
  """
  Represents a single point in the point cloud. Users should only call the
  functions visible here.
  """
  def __init__(self):
    super().__init__(self)

  @property
  @abstractmethod
  def parity(self) -> bool:
    pass
  
  @property
  @abstractmethod
  def channel(self) -> int:
    pass

  # TODO: This might be better as a parameter to the layer's ForwardPass
  # functions.
  @property
  @abstractmethod
  def vector(self): # Unsure what this type should be
    pass

  @abstractmethod
  def getNeighbors(self, distance : float) -> List[Point]:
    pass

  @abstractmethod
  def getOtherChannels(self) -> List[Point]:
    pass
  
