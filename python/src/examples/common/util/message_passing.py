from torch_geometric.nn import MessagePassing as PygMessagePassing

from examples.common.impl.util.internal_caller import InternalCaller
from src.examples.common.impl.point_cloud_base import PointCloudBase
from examples.common.impl.pyg.pyg_point_cloud_aggregator import PygPointCloudAggregator

class MessagePassing(InternalCaller, PygMessagePassing):
    """
    A wrapper around the Pytorch Geometric MessagePassing class that handles
    interactions between PyG and GElib, and any conversions through invalid
    PointCloud states.
    """
    def __init__(self, *args, **kwargs):
        # Switch out the aggregator.
        if not 'aggr' in kwargs:
            kwargs['aggr'] = 'sum'
        kwargs['aggr'] = PygPointCloudAggregator(kwargs['aggr']) 
        
        self.has_initialized_ = False

        # And propegate.
        super().__init__(
            internal_methods = ['translateFromPyg', 'translateToPyg'],
            *args,
            **kwargs)

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
            if name == "propegate":
                def newpropegate(*args, **kwargs):
                    key = 'x'
                    if key in kwargs:
                        kwargs[key] = self.translateToPyg(kwargs[key])
                    result = attr(*args, **kwargs)
                    assert self.translateFromPyg(result)
                return newpropegate
            
            elif name == "message":
                def newmessage(*args, **kwargs):
                    for key in ['x_i', 'x_j']:
                      if key in kwargs:
                          x = kwargs[key]
                          if x == None:
                              continue
                          kwargs[key] = self.translateFromPyg(x)
                    result = attr(*args, **kwargs)
                    assert isinstance(result, PointCloudBase)
                    return self.translateToPyg(result)
                return newmessage
        
        return attr
    
    def translateFromPyg(self, instance):
        assert isinstance(instance, PointCloudBase)
        return instance.FromPygPropegationFormat()

    def translateToPyg(self, instance):
        assert isinstance(instance, PointCloudBase)
        return instance.ToPygPropegationFormat()