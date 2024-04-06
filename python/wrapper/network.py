from abc import abstractmethod, property
from typing import Any, Callable, Generic, List, TypeVar

from gelib import Layer, Location, NetworkImpl

class PointConfig:
  """
  Specifies how points should be created, and what properties should be used
  to determine whether two points are unique.
  """
  def __init__(self, use_parity = False):
    self.use_parity = use_parity

class NetworkConfig:
  """
  The configuration to be used for the network
  """
  def __init__(self, point_locations : List[Location]):
    self.locations = point_locations

class Network:
  """
  The publically accessible parts of a Network. Users should only call the
  functions visible here, and should create a new instance using CreateNetwork()
  below.
  """
  
  @abstractmethod
  def doForwardPass(self):
    pass

  @abstractmethod
  def doBackwardPass(self):
    pass

def createNetwork(layers : List[Layer],
                  network_config : NetworkConfig,
                  point_config : PointConfig = PointConfig()) -> Network:
  return NetworkImpl(layers, network_config, point_config)