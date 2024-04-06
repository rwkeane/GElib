from typing import Any, Callable, Generic, List, TypeVar

from gelib import gather, calculateCGProduct, createNetwork, Layer, Network, \
                  Point

class TensorFieldNetworkCGLayer(Layer):
  def __init__(self, distance_cutoff : float):
    self.distance_cutoff_ = distance_cutoff
    super().__init__()

  def InitializePoint(self, point : Point):
    neighbors = point.getNeighbors(self.distance_cutoff_)
    for neighbor in neighbors:
      point.createWeight(neighbor, kDefaultValue) # kDefaultValue TBD

  def forwardPassForVertices(self, point : Point):
    return gather(sum,
                  point,
                  point.getNeighbors(self.distance_cutoff_),
                  calculateCGProduct,
                  point)
  
class TensorFieldNetworkSelfInteractionLayer(Layer):
  def __init__(self):
    super().__init__()

  def InitializePoint(self, point : Point):
    other_channels = point.getOtherChannels()
    for channel_point in other_channels:
      point.createWeight(channel_point, kDefaultValue) # kDefaultValue TBD

  def forwardPassForVertices(self, point : Point):
    return gather(sum,
                  point,
                  point.getOtherChannels(),
                  identity, # Does nothing
                  point)
  
if __name__ == "__main__":
    # 1. Define config for network.
    

    # 2. Create new layers


    # 3. Use createNetwork() to create a network.
    

    # 4. Forward pass, backward pass, repeat.

    pass