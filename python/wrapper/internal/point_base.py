from abc import abstractmethod, property
import bisect
from typing import List

from gelib import LayerBase, Location, Point

class PointBase(Point):
  """
  Partial implementation of a point. Excludes information that is specific to
  a single layer, such as the channel.
  """
  def __init__(self, location : Location, parity : bool):
    super().__init__(self)
    self.location_ : Location = location
    self.parity_ : bool = parity

  @property
  def parity(self) -> bool:
    return self.parity_
  
  @property
  def channel(self) -> int:
    assert False, "This should not be called in PointBase!"
    return 0

  @property
  def vector(self): # Unsure what this type should be
    pass

  @property
  def location(self): # Unsure what this type should be
    return self.location_

  def distance(self, other_point):
    return self.location().distance(other_point.location())