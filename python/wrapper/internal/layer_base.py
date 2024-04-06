from abc import abstractmethod, property
from typing import Any, Dict, List, TypeVar, Generic

from gelib import PointBase, PointImpl, WeightRegistry

class LayerBase(WeightRegistry):
  def __init__(self, channel_count):
    self.channel_count_ : int = channel_count
    self.points_ : List[PointImpl] = []
    super().__init__(self)

  def getAllPoints(self) -> List[PointImpl]:
    return self.points_

  def assignPoints(self, point_bases : List[PointBase]):
    for point_base in point_bases:
      for channel in range(self.channels_):
        self.points_.append(PointImpl(point_base, channel))