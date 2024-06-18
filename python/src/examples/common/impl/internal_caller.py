from typing import List, Optional

from examples.common.impl.internal_type import InternalType

class InternalCaller(object):
    def __init__(self,
                 internal_methods : Optional[List[str]] = None,
                 *args,
                 **kwargs):
        super(*args, **kwargs)

        if internal_methods == None:
            internal_methods = [ 'forward' ]
        self.internal_methods_ = internal_methods

    def __getattribute__(self,name):
        """
        Set all |TrackedType| instances to 
        """
        attr = super().__getattribute__(self, name)
        if hasattr(attr, '__call__') and name in self.internal_methods_:
            def newCall(*args, **kwargs):
                tracked_types : List[InternalType] = []
                for i, arg in enumerate(args):
                    if isinstance(arg, InternalType):
                        tracked_types.append(arg)
                        
                for key, value in kwargs.items():
                    if isinstance(value, InternalType):
                        tracked_types.append(value)

                for instance in tracked_types:
                    instance.setInternal()

                result = attr(*args, **kwargs)

                for instance in tracked_types:
                    instance.unsetState()

                return result
            return newCall
        else:
            return attr