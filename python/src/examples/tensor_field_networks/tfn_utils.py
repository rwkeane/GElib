import math
from functools import partial
from typing import Any, Callable, Generic, List, TypeVar

import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

def createGraphData(positions : torch.tensor,
                    max_dist : float,
                    num_input_channels : int,
                    default_values : torch.tensor = None) -> Data:
  # default_values of size |point count| x |channels| x (inner dimensions)
  assert positions != None
  assert default_values != None

  data_point_count = len(positions)
  assert len(default_values) == data_point_count

  # Creates a complete graph
  edge_list = []
  for i in range(data_point_count):
    assert len(positions[i]) == num_input_channels

    for j in range(data_point_count):
      if i == j: 
        continue
      
      if (max_dist == None or 
          math.dist(positions[i], positions[j]) <= max_dist):
        edge_list.append([i, j])

  edge_index = torch.tensor(edge_list, dtype = torch.int32)
  data = Data(x = default_values, edge_index = edge_index.t().contiguous())

  data.validate(raise_on_error = True)
  return data

def createGraphDataFactory(positions : torch.tensor,
                           max_dist : float = None):
  return partial(createGraphData, positions, max_dist)