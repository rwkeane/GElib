from abc import abstractmethod, property
from typing import Any, Dict, TypeVar, Generic

from gelib import Point, LayerBase

TResultType = TypeVar('TResultType')
class Layer(LayerBase, Generic[TResultType]):
  """
  Represents a single layer of the network. A network will consist of a series
  of layers and an associated ordering.
  """
  def __init__(self, channel_count):
    super().__init__(self, channel_count)

  def InitializePoint(self, point : Point):
    """
    To be overridden by children that need to do per-point initialization, such
    as creating a weight assignment for each point.
    """
    assert point != None

  def forwardPassForVertices(self, point : Point) -> TResultType:
    """
    Performs message aggregation for |point|.. Must be implemented to support
    message passing on the vertices
    """
    return None

  def forwardPassForEdges(self, point1 : Point, point2 : Point) -> TResultType:
    """
    Performs aggregation across the edge from |point1| to |point2|. Must be
    implemented to support message passing over edges
    """
    return None



  