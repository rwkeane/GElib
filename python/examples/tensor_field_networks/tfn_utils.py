import math
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Any, Callable, Generic, List, TypeVar

def createGraphData(default_values : torch.tensor,
                    positions : torch.tensor,
                    num_input_channels : int,
                    max_dist : float = None):
  # default_values of size |point count| x |channels| x (inner dimensions)
  assert len(positions) == len(default_values)

  # Creates a complete graph
  edge_list = []
  for i in range(len(default_values)):
    assert len(default_values[i]) == num_input_channels

    for j in range(len(default_values)):
      if i == j: 
        continue
      
      if (max_dist == None or 
          math.dist(positions[i], positions[j]) <= max_dist):
        edge_list.append([i, j])

  edge_index = torch.tensor(edge_list, dtype=torch.int32)
  data = Data(x=default_values, edge_index=edge_index.t().contiguous())

  data.validate(raise_on_error=True)
  return data