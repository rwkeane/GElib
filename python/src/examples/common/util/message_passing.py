from functools import partial
from typing import Any, Dict, List, Optional, Set, Union
import torch
from torch_geometric.nn import MessagePassing as PygMessagePassing

from src.examples.common.impl.util.internal_caller import InternalCaller
from src.examples.common.impl.point_cloud_base import PointCloudBase
from src.examples.common.impl.pyg.pyg_point_cloud_aggregator import \
    PygPointCloudAggregator

class MessagePassing(InternalCaller, PygMessagePassing):
    """
    A wrapper around the Pytorch Geometric MessagePassing class that handles
    interactions between PyG and GElib, and any conversions through invalid
    PointCloud states.

    NOTE: This class really reaches into the internals of PyG's MessagePassing
    class. It's possible (albeit unlikely) that a future update to PyG will
    break it.
    """
    def __init__(self, *args, **kwargs):
        # Switch out the aggregator.
        if not 'aggr' in kwargs:
            kwargs['aggr'] = 'sum'
        kwargs['aggr'] = PygPointCloudAggregator(kwargs['aggr']) 
        
        self.has_initialized_ = False

        # And propegate.
        internal_methods = \
            ['translateFromPyg', 'translateToPyg', 'pre_hook', 'post_hook']
        super().__init__(internal_methods = internal_methods, *args, **kwargs)
        
        # Register hooks to modify all PointCloud instances received as args.
        #
        # Pre-Hooks.
        self.register_propagate_forward_pre_hook(
            partial(MessagePassing.pre_hook, self._propagate_forward_pre_hooks))
        self.register_message_forward_pre_hook(
            partial(MessagePassing.pre_hook, self._message_forward_pre_hooks))
        self.register_aggregate_forward_pre_hook(
            partial(MessagePassing.pre_hook, self._aggregate_forward_pre_hooks))
        self.register_message_and_aggregate_forward_pre_hook(
            partial(MessagePassing.pre_hook,
                    self._message_and_aggregate_forward_pre_hooks))
        self.register_edge_update_forward_pre_hook(
            partial(MessagePassing.pre_hook,
                    self._edge_update_forward_pre_hooks))
        
        # Post hooks.
        self.register_propagate_forward_hook(
            partial(MessagePassing.post_hook, self._propagate_forward_hooks))
        self.register_message_forward_hook(
            partial(MessagePassing.post_hook, self._message_forward_hooks))
        self.register_aggregate_forward_hook(
            partial(MessagePassing.post_hook, self._aggregate_forward_hooks))
        self.register_message_and_aggregate_forward_hook(
            partial(MessagePassing.post_hook,
                    self._message_and_aggregate_forward_hooks))
        self.register_edge_update_forward_hook(
            partial(MessagePassing.post_hook, self._edge_update_forward_hooks))

        # Don't set this until the end to save a bit of time.
        self.has_initialized_ = True

    def __getattribute__(self, name):
        """
        Modify the parameters to and from the propegate() and message()
        functions. 
        """
        attr = super().__getattribute__(name)

        # Only use this special case after super().__init__() has completed, as
        # it inspects the attributes of the child class to find message(), which
        # fails if its overridden here.
        if hasattr(attr, '__call__') and self.has_initialized_:
            if name == "propagate":
                def newpropegate(*args, **kwargs):
                    key = 'x'
                    if key in kwargs:
                        kwargs[key] = self.translateToPyg(kwargs[key])
                    else:
                        assert False
                    result = attr(*args, **kwargs)
                    assert self.translateFromPyg(result)
                return newpropegate
            
            elif name == "message":
                def newmessage(*args, **kwargs):
                    for key in ['x_i', 'x_j']:
                       if key in kwargs:
                           x = kwargs[key]
                           assert not x is None, x
                           kwargs[key] = self.translateFromPyg(x)
                    result = attr(*args, **kwargs)
                    assert isinstance(result, PointCloudBase)
                    as_pyg = self.translateToPyg(result)
                    return as_pyg
                return newmessage
        
        return attr
    
    def _collect(
        self,
        args: Set[str],
        edge_index: torch.Tensor,
        size: List[Optional[int]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        out = super()._collect(args, edge_index, size, kwargs)

        for arg in args:
            if arg[-2:] not in ['_i', '_j']:
                continue

            key = arg[:-2]
            val = kwargs[key]
            if isinstance(val, PointCloudBase):
                results = []
                for part in val.data().parts:
                    current_kwargs = kwargs.copy()
                    current_kwargs[key] = part
                    current_result = \
                        super()._collect(args, edge_index, size, current_kwargs)
                    new_tensor = current_result[arg]
                    assert new_tensor != None
                    assert isinstance(new_tensor, torch.Tensor)
                    results.append(new_tensor)
                new_cloud = val.CloneWithNewValue(results)
                assert new_cloud != None
                out[arg] = new_cloud

        return out
    
    def pre_hook(hook_list, module, all_args):
        all_args = list(all_args)
        modified = False
        for i in range(len(all_args)):
            arg = all_args[i]

            # This might be the kwargs object.
            #
            # TODO: This may cause issues if multiple parameters of the input
            # are PointCloudBase instances, but in practice that doesn't seem to
            # happen.
            if isinstance(arg, dict):
                for key, val in arg.items():
                    if isinstance(val, PointCloudBase):
                        modified = True
                        results = []
                        for part in val.data().parts:
                            current_args = all_args.copy()
                            current_dict = current_args[i]
                            current_dict[key] = part
                            for hook in hook_list.values():
                                res = hook(module, tuple(current_args))
                                if res is not None:
                                    current_args = res
                            new_tensor = (current_args[i])[key]
                            assert isinstance(new_tensor, torch.Tensor)
                            results.append(new_tensor)
                        arg[key] = val._createObject(results)
        
        if modified:
            return tuple(all_args)
        else:
            return None

    def post_hook(hook_list, module, all_args, out):
        if isinstance(out, PointCloudBase):
            results = []
            for part in out.data().parts:
                current_out = part
                for hook in hook_list.values():
                    res = hook(module, all_args, current_out)
                    if res is not None:
                        current_out = res
                results.append(current_out)
            return out._createObject(results)
        
        return None
    
    def translateFromPyg(self, instance):
        assert isinstance(instance, PointCloudBase)
        return instance.FromPygPropegationFormat()

    def translateToPyg(self, instance):
        assert isinstance(instance, PointCloudBase)
        return instance.ToPygPropegationFormat()