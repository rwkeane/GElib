from abc import abstractmethod, property
from typing import List

from gelib import Point, WeightRegistry

class Point(WeightRegistry):
  def __init__(self):
    super().__init__(self)

  @property
  @abstractmethod
  def parity() -> bool:
    pass
  
  @property
  @abstractmethod
  def channel() -> int:
    pass

  @property
  @abstractmethod
  def vector(): # Unsure what this type should be
    pass

  @abstractmethod
  def getNeighbors(distance : float) -> List[Point]:
    pass

  @abstractmethod
  def getOtherChannels() -> List[Point]:
    pass
  
