from typing import Callable, Generic, List, TypeVar

from gelib import Point, WeightRegistry

TAggType = TypeVar('TAggType')
def gather(aggregation_function : Callable[[List[TAggType]], TAggType],
           origin_point : Point,
           points_to_aggregate_over : List[Point],
           process_points : Callable[[Point, Point], TAggType],
           weight_registry : WeightRegistry) -> TAggType:
  values = []
  for point in points_to_aggregate_over:
    new_value = process_points(origin_point, point)

    first_weight = weight_registry.getWeight(origin_point)
    second_weight = weight_registry.getWeight(point)
    if (first_weight != None):
      new_value *= first_weight
    if (second_weight != None):
      new_value *= second_weight

    values.append(new_value)

  return aggregation_function(values)