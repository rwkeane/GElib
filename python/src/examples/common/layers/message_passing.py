from torch_geometric.nn import MessagePassing as PygMessagePassing

from src.examples.common.impl.internal_caller import InternalCaller
from src.examples.common.impl.point_cloud_base import PointCloudBase
from src.examples.common.impl.point_cloud_aggregator import PointCloudAggregator

class MessagePassing(InternalCaller, PygMessagePassing):
    """
    A wrapper around the Pytorch Geometric MessagePassing class that handles
    interactions between PyG and GElib, and any conversions through invalid
    PointCloud states.
    """
    def __init__(self, *args, **kwargs):
        if not 'aggr' in kwargs:
            kwargs['aggr'] = 'sum'

        kwargs['aggr'] = PointCloudAggregator(kwargs['aggr']) 
        
        super().__init__(*args, **kwargs)

    def __getattribute__(self,name):
        """
        Modify the parameters to and from the propegate() and message()
        functions. 
        """
        attr = super(self, InternalCaller).__getattribute__(self, name)

        if hasattr(attr, '__call__'):
            if name == "propegate":
                def newpropegate(*args, **kwargs):
                    key = 'x'
                    if key in kwargs:
                        x = kwargs[key]
                        assert isinstance(x, PointCloudBase)
                        kwargs[key] = x.ToPygPropegationFormat()
                    result = attr(*args, **kwargs)
                    assert isinstance(result, PointCloudBase)
                    return result.FromPygPropegationFormat()
                return newpropegate
            elif name == "message":
                def newmessage(*args, **kwargs):
                    for key in ['x_i', 'x_j']:
                      if key in kwargs:
                          x = kwargs[key]
                          assert isinstance(x, PointCloudBase)
                          kwargs[key] = x.FromPygPropegationFormat()
                    result = attr(*args, **kwargs)
                    assert isinstance(result, PointCloudBase)
                    return result.ToPygPropegationFormat()
                return newmessage
        else:
            return attr