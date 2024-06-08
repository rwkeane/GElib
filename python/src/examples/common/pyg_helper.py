from typing import Tuple, List, Iterable
import torch

from ...gelib import SO3partArr

def reshapeInputForPyg(input : SO3partArr) -> SO3partArr:
  assert isinstance(input, SO3partArr), type(input)

  # input is (batch, channel, 2l + 1, N)
  # Output is (N, batch, channel, 2l + 1)
  order = tuple(range(-1, input.dim() - 1, 1))
  assert len(order) == input.dim(), "{0} for {1}-dim".format(order, input.dim())
  return input.permute(order).contiguous()

def undoReshapeInputForPyg(input : SO3partArr) -> SO3partArr:
  assert isinstance(input, SO3partArr), type(input)

  # input is (batch, channel, 2l + 1, N)
  # Output is (N, batch, channel, 2l + 1)
  order = list(range(1, input.dim(), 1))
  order.append(0)
  order = tuple(order)
  assert len(order) == input.dim(), "{0} for {1}-dim".format(order, input.dim())
  return input.permute(order).contiguous()

def flattenForPygPropegate(input : SO3partArr) -> Tuple[SO3partArr, Tuple]:
  size = input.size()
  return input.view(size[0], -1), size

def undoFlattenForPygPropegate(
    input : SO3partArr, size : Iterable) -> SO3partArr:
  if not isinstance(size, List):
    size = list(size)
  size[0] = input.size()[0]
  return input.view(size)