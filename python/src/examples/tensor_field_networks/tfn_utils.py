import math
from functools import partial
from typing import Any, Callable, Generic, List, TypeVar

import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from ...gelib import SO3partArr

from src.examples.tensor_field_networks.pyg_helper import \
  reshapeInputForPyg, undoReshapeInputForPyg

def createGraphData(positions : torch.Tensor,
                    values : SO3partArr,
                    max_dist : float = None) -> Data:
  # default_values of size |point count| x |channels| x (inner dimensions)
  assert positions != None
  assert values != None

  data_point_count = positions.size()[0]
  distances = torch.cdist(positions, positions)
  assert distances.dim() == 2
  # assert values.size()[0] == data_point_count, values.size()

  # Creates a complete graph
  edge_list = []
  for i in range(data_point_count):
    for j in range(data_point_count):
      if i == j: 
        continue
      
      if (max_dist == None or 
          math.dist(positions[i], positions[j]) <= max_dist):
        edge_list.append([i, j])

  edge_index = torch.tensor(edge_list, dtype = torch.int64)
  data = Data(x = values, edge_index = edge_index.t().contiguous())
  data.point_distances = distances
  data.point_positions = positions

  # Don't reshape input for ALL layers, only for the PyG layers which must do it
  # manually in their forward() call.
  data.x = reshapeInputForPyg(data.x)
  data.validate(raise_on_error = True)
  data.x = undoReshapeInputForPyg(data.x)

  return data

def createOnesTensor(l : int, size : int):
  ones = SO3partArr.ones(1, [1], l, size)
  b = ones.size()[0]
  l_out = ones.size()[-2]
  n = ones.size()[-1]
  assert ones.getb() == b
  assert ones.getn() == n
  assert b == 1 and l_out == 2 * l + 1 and n == size
  return ones