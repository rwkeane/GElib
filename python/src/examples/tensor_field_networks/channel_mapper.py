import math
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential

class ChannelMapper(Linear):
  """
  Maps from |in_channels| to |out_channels| using a linear layer. Last dimension
  """
  def __init__(self, in_channels, out_channels, bias):
    self.in_channels_ = in_channels
    self.out_channels_ = out_channels

    super().__init__(in_channels, out_channels, bias)

  def forward(self, x : torch.Tensor):
    shape = x.size()
    shape[-1] = self.out_channels_

    result = super().forward(x.view(-1, self.in_channels_))

    return result.view(shape)