from typing import Tuple, List, Iterable
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from ...gelib import SO3partArr

def updateMessagePassingInputs(edge_index : torch.Tensor, 
                               input : SO3partArr):
  assert isinstance(input, SO3partArr)

  # Cone the edge index and 
  assert edge_index.dim() == 2 and edge_index.size()[1] == 2, edge_index.size()
  length = edge_index.size()[0]
  channel_count = input.size()[-1]
  to_add = torch.stack([rep // length for rep in range(length * channel_count)])
  to_add = to_add.unsqueeze(-1).repeat(1,2)
  cloned_index = edge_index.repeat((channel_count,1)) + to_add

def reshapeInputForPyg(input : SO3partArr) -> SO3partArr:
  assert isinstance(input, SO3partArr), type(input)

  # input is (batch, channel, 2l + 1, N)
  # Output is (N, batch, channel, 2l + 1)
  order = tuple(range(-1, input.dim() - 1, 1))
  assert len(order) == input.dim(), "{0} for {1}-dim".format(order, input.dim())
  return input.permute(order)

def undoReshapeInputForPyg(input : SO3partArr) -> SO3partArr:
  assert isinstance(input, SO3partArr), type(input)

  # input is (batch, channel, 2l + 1, N)
  # Output is (N, batch, channel, 2l + 1)
  order = list(range(1, input.dim(), 1))
  order.append(0)
  order = tuple(order)
  assert len(order) == input.dim(), "{0} for {1}-dim".format(order, input.dim())
  return input.permute(order)

def flattenForPygPropegate(input : SO3partArr) -> Tuple[SO3partArr, Tuple]:
  size = input.size()
  return input.view(size[0], -1), size

def undoFlattenForPygPropegate(input : SO3partArr, size : Iterable) -> SO3partArr:
  if not isinstance(size, List):
    size = list(size)
  size[0] = input.size()[0]
  return input.view(size)