
import math
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from typing import Any, Callable, Generic, List, TypeVar

class SelfInteractionLayer(Module):
    def __init__(self,
                 in_channels : int,
                 out_channels : int):
        super().__init__()

        self.lin = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        # Linearly transform node features
        x = self.lin(x)

        # Sum across channels
        out = torch.sum(x, dim=1, keepdim=True)

        return out