from enum import Enum
from typing import Any, Dict, TypeVar, Generic

TRegistryValue = TypeVar('TRegistryValue')
class WeightRegistry(Generic[TRegistryValue]):
  """
  This class is responsible for maintaining the weights assocaited with the
  parent instance. State is maintained as "Initializing" and "Running", where
  new weights can only be created during the former.
  """
  class State(Enum):
    kInitializing = 1
    kRunning = 2

  def __init__(self):
    self.weight_assignments_ : Dict[Any, TRegistryValue] = {}
    self.state_ = WeightRegistry.State.kInitializing

  def setRunning(self):
    self.state_ = WeightRegistry.State.kRunning

  def createWeight(self, key, value : TRegistryValue):
    """
    Creates a new weight associated with key |key|. Cannot be called on keys
    which already have a value assigned.
    """
    assert self.state_ == WeightRegistry.State.kInitializing 
    assert key != None
    assert value != None

    assert(self.weight_assignments_[key] == None)
    self.weight_assignments_ = value

  def getWeight(self, key) -> TRegistryValue:
    """
    Gets the weight associated with previously created key. May only be called
    on valid keys.
    """
    assert key != None

    value = self.weight_assignments_[self]
    assert value != None
    return value