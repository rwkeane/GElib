from typing import Any, Callable, Generic, List, TypeVar

from gelib import Layer, Network, NetworkConfig, PointBase, PointConfig

class NetworkImpl(Network):
  """
  Implementation of the publicly visible Network class. Hides all of the
  internals from the users so they can't mess up the internal state.
  """
  def __init__(self,
               layers : List[Layer],
               network_config : NetworkConfig,
               point_config : PointConfig = PointConfig()):
    """
    Creates a new network consisting of |layers| which will be processed in
    order. Points will be created according to |network_config|.
    """
    self.layers_ : List[Layer] = layers

    self.points_ : List[PointBase] = []
    parities : List[bool] = [ True ]
    if point_config.use_parity:
      parities.append(False)

    # Create all points.
    for location in network_config.locations:
      for parity in parities:
        self.points_.append(PointBase(location, parity))
    for layer in self.layers_:
      layer.assignPoints(self.points_)

    # Initialize all points.
    for point in self.points_:
      for layer in self.layers_:
        layer.InitializePoint(point)

    # Set points and layers as running so no more weights get created.
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