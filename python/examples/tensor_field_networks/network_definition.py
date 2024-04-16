import math
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from typing import Any, Callable, Generic, List, TypeVar

import gelib

class TensorFieldNetwork(Module):
  def __init__(self, )