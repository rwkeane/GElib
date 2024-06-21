from typing import Tuple, List, Iterable

from gelib import SO3vecArr

from src.examples.common.impl.point_cloud_base import PointCloudBase
from src.examples.common.point_cloud import PointCloud

def reshapeInputForPyg(input : PointCloud) -> PointCloud:
  assert isinstance(input, PointCloud), type(input)
  assert input.dim() >= 3


  # input is (batch, channel, vertices, 2l + 1, N)
  # Output is (vertices, 2l + 1, N, batch, channel)
  order = tuple(range(-3, input.dim() - 3, 1))
  assert len(order) == input.dim(), "{0} for {1}-dim".format(order, input.dim())
  return input.permute(order).contiguous()

def undoReshapeInputForPyg(input : PointCloud) -> PointCloud:
  assert isinstance(input, PointCloud), type(input)
  assert input.dim() >= 3

  # input is (vertices, 2l + 1, N, batch, channel)
  # Output is (batch, channel, vertices, 2l + 1, N)
  order = list(range(3, input.dim(), 1)) + [ 0, 1, 2 ]
  order = tuple(order)
  assert len(order) == input.dim(), "{0} for {1}-dim".format(order, input.dim())
  return input.permute(order).contiguous()

def flattenForPygPropegate(input : PointCloudBase) -> Tuple[PointCloud, Tuple]:
  size = input.allSizes()
  return input.view((size[0])[0], -1), size

def undoFlattenForPygPropegate(
    input : PointCloudBase, size : Iterable) -> PointCloud:
  new_sizes = []
  for old_size in size:
    old_size = list(old_size)
    old_size[0] = input.size()[0]
    new_sizes.append(old_size)
  return input.allViews(new_sizes)

