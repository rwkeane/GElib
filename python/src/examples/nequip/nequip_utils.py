import math

from ...gelib import SO3partArr

kPositive = 1
kNegative = 0

def createOnesTensor(l : int, size : int, channels = 1):
  # 2 is the parities
  ones = SO3partArr.ones(1, [2, channels], l, size)
  b = ones.size()[0]
  l_out = ones.size()[-2]
  n = ones.size()[-1]
  assert ones.getb() == b
  assert ones.getn() == n
  assert b == 1 and l_out == 2 * l + 1 and n == size
  return ones