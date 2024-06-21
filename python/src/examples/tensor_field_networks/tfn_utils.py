import math
from functools import partial
from typing import Any, Callable, Generic, List, TypeVar

import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from gelib import SO3vecArr

def createOnesTensor(l : int, size : int, channels : int = 1, batch : int = 1):
  tau = [1 for _ in range(l + 1)]
  ones = SO3vecArr.ones(batch, [channels, size], tau)

  # Validate.
  assert ones.getb() == 1
  if __debug__:
    tau = ones.tau()
    for t in tau:
        assert t == 1
        
  return ones