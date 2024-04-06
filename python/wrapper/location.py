import math

class Location:
  def __init__(self, x : float, y : float, z : float):
    self.x_ = x
    self.y_ = y
    self.z_ = z

  def distance(self, other) -> float:
    x_part = self.x_ - other.x_
    y_part = self.y_ - other.y_
    z_part = self.z_ - other.z_
    return math.sqrt(x_part * x_part + y_part * y_part + z_part * z_part)