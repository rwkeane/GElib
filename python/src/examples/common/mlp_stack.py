from typing import Callable, TypeVar, Generic
import torch
from torch.nn import Linear, Module, ModuleList, Parameter

TMlpType = TypeVar('TMlpType') 
class MlpStack(Generic[TMlpType], Module):
    """
    A 2D array of MLPs, as created by |mlp_factory|. The input is assumed to
    have |channels| channels and |l_max| to be the maximum supported l value.
    """
    def __init__(self,
                 channels : int,
                 l_max : int,
                 mlp_factory : Callable[[TMlpType]]):
        assert mlp_factory != None
        assert isinstance(mlp_factory, Callable[[TMlpType]])
        
        super(Module, self).__init__()

        mlps = torch.nn.ModuleList([])
        for channel in range(channels):
            channel_list = torch.nn.ModuleList([])
            for l in range(2 * l_max + 1):
                channel_list.append(mlp_factory())
            mlps.append(channel_list)
        self.mlps_ = mlps
        
    def reset_parameters(self):
        for layer in self.mlps_:
            if isinstance(layer, torch.nn.ModuleList):
                self.resetMlpStack(layer)
                continue

            assert isinstance(layer, TMlpType)
            layer.reset_parameters()
    
    def forward(self, x : torch.Tensor):
        return torch.stack([
            torch.stack([mlp(x) for mlp in mlp_list], dim=-1) \
                for mlp_list in self.mlps_
        ])