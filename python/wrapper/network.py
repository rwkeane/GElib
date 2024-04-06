from typing import Any, Callable, Generic, List, TypeVar

from gelib import Layer, Point

class PointConfig:
  """
  Specifies how points should be created, and what properties should be used
  to determine whether two points are unique.
  """
  def __init__(self, use_parity = False):
    self.use_parity = False

class Network:
  def __init__(self, layers : List[Layer],
               network_configuration,
               point_config : PointConfig = PointConfig()):
    self.layers_ : List[Layer] = layers

    self.points_ : List[Point] = []
    # Create points based on config

    # Initialize all points
    for point in self.points_:
      for layer in self.layers_:
        layer.InitializePoint(point)

    for point in self.points_:
      point.setRunning()

    for layer in self.layers_:
      layer.setRunning()

  def doForwardPass(self):
    for layer in self.layers_:
      for point in self.points_:
        layer.forwardPassForVertices(point)
        for point2 in self.points_:
          if point == point2:
            continue

          layer.forwardPassForEdges(point, point2)

  def doBackwardPass(self):
    # TODO
    pass