import math
from typing import Optional, Callable

import torch
import torch.nn as nn

from ...gelib import SO3vecArr

class NequIPOutputBlock(nn.Module):
    def __init__(self,
                 input_dim : int,
                 embedding_dim: int,
                 hidden_features : Optional[int] = None,
                 nonlinearity : Callable[[torch.Tensor], torch.Tensor] = None):
        super().__init__()

        if input_dim == None:
            hidden_features = int(math.sqrt(input_dim))
        if nonlinearity == None:
            nonlinearity = torch.nn.functional.silu

        self.embedding_dim_ = embedding_dim
        self.embedding_linear_ = nn.Linear(embedding_dim, 1, bias = True)
        self.nonlinearity_ = nonlinearity
        self.self_interaction_1_ = \
            nn.Linear(input_dim, hidden_features, bias = True)
        self.self_interaction_2_ = \
            nn.Linear(hidden_features, 1, bias = True) 

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_linear_.reset_parameters()
        self.self_interaction_1_.reset_parameters()
        self.self_interaction_2_.reset_parameters()

    def forward(self, x: SO3vecArr):
        assert isinstance(x, SO3vecArr), type(x)
        assert x.dim() >= 5, x.size()

        # Intput x shape: 
        # (batch, ..., embedding length, parity, channels, l, atoms)
        # After: (batch, ..., channels, atoms, l = 0, embedding_length)
        x = x[...,0,:,0,:].squeeze(-4).transpose(-1, -4)
        
        # Flatten and eliminate the embedding dim.
        previous_size = x.size()
        x = x.reshape(-1, previous_size[-1])
        x = self.embedding_linear_.forward(x)

        # Undo Reshape.
        # After: (batch, ..., channels, l = 0, atoms)
        new_size = list(previous_size)
        new_size[-1] = 1
        x = x.reshape(new_size).squeeze(-1).transpose(-1, -2)

        # Do it all again for the atom dim.
        # input: (batch, ..., channels, l = 0, atoms)
        # After: (-1, atoms)
        previous_size = x.size()
        x = x.reshape(-1, previous_size[-1])
        x = self.self_interaction_1_.forward(x)

        # Undo Reshape.
        new_size = list(previous_size)
        new_size[-1] = 1
        atomic_energies = x.reshape(new_size)

        # Reshape back to match SO3vecArr structure
        assert atomic_energies.getl() == 0
        return atomic_energies