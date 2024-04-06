from abc import abstractmethod, property
from typing import List

from gelib import Point, WeightRegistry

class Point(WeightRegistry):
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
  
