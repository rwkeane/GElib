import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential

class ConcatenationLayer(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Concatenate along channel axis
        # [N, channels, M]
        return torch.cat(x, dim=2)