import math
from typing import Tuple, List, Iterable
import torch
from torch_geometric.data import Data

from ...gelib import SO3vecArr

def reshapeInputForPyg(input : SO3vecArr) -> SO3vecArr:
  assert isinstance(input, SO3vecArr), type(input)

  # input is (batch, channel, 2l + 1, N)
  # Output is (N, batch, channel, 2l + 1)
  order = tuple(range(-1, input.dim() - 1, 1))
  assert len(order) == input.dim(), "{0} for {1}-dim".format(order, input.dim())
  return input.permute(order).contiguous()

def undoReshapeInputForPyg(input : SO3vecArr) -> SO3vecArr:
  assert isinstance(input, SO3vecArr), type(input)

  # input is (batch, channel, 2l + 1, N)
  # Output is (N, batch, channel, 2l + 1)
  order = list(range(1, input.dim(), 1))
  order.append(0)
  order = tuple(order)
  assert len(order) == input.dim(), "{0} for {1}-dim".format(order, input.dim())
  return input.permute(order).contiguous()

def flattenForPygPropegate(input : SO3vecArr) -> Tuple[SO3vecArr, Tuple]:
  size = input.size()
  return input.view(size[0], -1), size

def undoFlattenForPygPropegate(
    input : SO3vecArr, size : Iterable) -> SO3vecArr:
  if not isinstance(size, List):
    size = list(size)
  size[0] = input.size()[0]
  return input.view(size)