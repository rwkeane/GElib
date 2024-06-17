import torch.nn as nn

from gelib import SO3partArr

class BPart(nn.Module):
    def __init__(self):
        super().__init__()

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: SO3partArr):
        assert isinstance(x, SO3partArr), type(x)