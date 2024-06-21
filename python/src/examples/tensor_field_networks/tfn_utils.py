from gelib import SO3vecArr

def createOnesTensor(l : int, size : int, channels : int = 1, batch : int = 1):
  tau = [1 for _ in range(l + 1)]
  ones = SO3vecArr.ones(batch, [channels, size], tau)

  # Validate.
  assert ones.getb() == 1
  if __debug__:
    tau = ones.tau()
    for t in tau:
        assert t == 1
        
  return ones