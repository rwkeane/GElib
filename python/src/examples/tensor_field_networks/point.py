from typing import Iterable, List, TypeVar
import math
import torch

TFeatureType = TypeVar("TFeatureType")
class Point:
  def __init__(self, position: torch.tensor, feature : TFeatureType):
    self.feature_ = feature

    # Don't expose this to keep equivariance.
    self.position_ = position

  @classmethod
  def FromArrays(cls, positions: torch.tensor, features: Iterable[TFeatureType]):
    assert features.dim() == 2
    assert positions.dim() == 1

    positions_ct, dimension = positions.shape
    features_ct = features.shape
    assert isinstance(features_ct, int)
    assert positions_ct == features_ct

    points = []
    for i in range(positions_ct):
      points.append(Point(positions[i], features[i]))
    
    return points
  
  def feature(self) -> TFeatureType:
    return self.feature_
  
  def distance(self, other : 'Point'):
    assert isinstance(other, Point)
    
    # TODO: Cache this value.
    dist = math.dist(self.position_, other.position_)
    return dist