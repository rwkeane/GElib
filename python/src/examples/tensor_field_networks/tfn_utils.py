import math
from functools import partial
from typing import Any, Callable, Generic, List, TypeVar

import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from ...gelib import SO3partArr

def createOnesTensor(l : int, size : int, channels = 1):
  ones = SO3partArr.ones(1, [channels], l, size)
  b = ones.size()[0]
  l_out = ones.size()[-2]
  n = ones.size()[-1]
  assert ones.getb() == b
  assert ones.getn() == n
  assert b == 1 and l_out == 2 * l + 1 and n == size
  return ones