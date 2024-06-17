from typing import Tuple, List, Iterable

from gelib import SO3vecArr

from src.examples.common.impl.point_cloud_base import PointCloudBase
from src.examples.common.point_cloud import PointCloud

def reshapeInputForPyg(input : PointCloud) -> PointCloud:
  assert isinstance(input, PointCloud), type(input)

  # input is (batch, channel, 2l + 1, N)
  # Output is (N, batch, channel, 2l + 1)
  order = tuple(range(-1, input.dim() - 1, 1))
  assert len(order) == input.dim(), "{0} for {1}-dim".format(order, input.dim())
  return input.permute(order).contiguous()

def undoReshapeInputForPyg(input : PointCloud) -> PointCloud:
  assert isinstance(input, PointCloud), type(input)

  # input is (batch, channel, 2l + 1, N)
  # Output is (N, batch, channel, 2l + 1)
  order = list(range(1, input.dim(), 1))
  order.append(0)
  order = tuple(order)
  assert len(order) == input.dim(), "{0} for {1}-dim".format(order, input.dim())
  return input.permute(order).contiguous()

def flattenForPygPropegate(input : PointCloudBase) -> Tuple[PointCloud, Tuple]:
  size = input.allSizes()
  return input.view((size[0])[0], -1), size

def undoFlattenForPygPropegate(
    input : PointCloudBase, size : Iterable) -> PointCloud:
  return input.allViews(size)

