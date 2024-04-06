from enum import Enum
from typing import Any, Dict, TypeVar, Generic

TRegistryValue = TypeVar('TRegistryValue')
class WeightRegistry(Generic[TRegistryValue]):
  class State(Enum):
    kInitializing = 1
    kRunning = 2

  def __init__(self):
    self.weight_assignments_ : Dict[Any, TRegistryValue] = {}
    self.state_ = WeightRegistry.State.kInitializing

  def setRunning(self):
    self.state_ = WeightRegistry.State.kRunning

  def createWeight(self, key, value : TRegistryValue):
    assert self.state_ == WeightRegistry.State.kInitializing 
    assert key != None
    assert value != None

    assert(self.weight_assignments_[key] == None)
    self.weight_assignments_ = value

  def getWeight(self, key) -> TRegistryValue:
    assert key != None

    value = self.weight_assignments_[self]
    assert value != None
    return value