import bisect
from typing import List

from gelib import Location, LayerBase, Point, PointBase

class PointImpl(Point):
  """
  Implementation of a point. Much of the functionality defers to a PointBase
  provided to the ctor, with only layer-specific details implemented here.
  """
  def __init__(self, base : PointBase, channel : int):
    self.base_ = base
    self.channel_ = channel
    self.points_by_distance_ = None
    self.points_differing_by_channel_ = None
    
    super().__init__(self)
  
  @property
  def channel(self) -> int:
    return self.channel_

  @property
  def parity(self) -> bool:
    return self.base_.parity()

  @property
  def vector(self):
    return self.base_.vector()

  @property
  def location(self) -> Location:
    return self.base_.location()

  def getNeighbors(self, max_dist : float) -> List[Point]:
    """
    Performs a binary search to find all points within the given maximum
    distance.
    """
    assert self.points_by_distance_ != None

    def search(other):
      if isinstance(other, float):
        return other
      assert isinstance(other, PointBase)
      return self.distance(other)
    
    index = bisect.bisect_right(self.points_by_distance_, max_dist, key=search)
    return self.points_by_distance_[:index]

  def getOtherChannels(self) -> List[Point]:
    assert self.points_differing_by_channel_ != None
    return self.points_differing_by_channel_

  def distance(self, other_point):
    return self.location().distance(other_point.location())
  
  def initialize(self, layer : LayerBase):
    """
    Initializes the point, fetching a list of all other points in the layer and
    pre-processing them such that getNeighbors() and getOtherChannels() can be
    computed quickly.
    """
    assert self.points_by_distance_ == None
    assert self.points_differing_by_channel_ == None

    self.points_by_distance_ : List[PointImpl] = \
        [x for x in layer.getAllPoints() if (x.channel() == self.channel())] 
    self.points_by_distance_.sort(key=self.distance)

    # TODO: Be sure this won't pick up too many / too few points.
    self.points_differing_by_channel_ : List[PointImpl] = \
        [x for x in layer.getAllPoints() if (x.base_ == self.base_)]