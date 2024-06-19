from typing import List, Optional

from examples.common.util.internal_type import InternalType

class InternalCaller:
    """
    Used to mark classes that are allowed to call equivariance-breaking
    operations on an InternalType, from all of that class's methods named in
    |internal_methods|.

    WARNING: Do NOT use this unless you REALLY know what you are doing.
    """
    def __init__(self,
                 internal_methods : Optional[List[str]] = None,
                 *args,
                 **kwargs):
        # If no internal method is given, assume forward().
        if internal_methods == None:
            internal_methods = [ 'forward' ]
        self.internal_methods_ = internal_methods

        # Use hashing for containment checks instead of list iteration for lists
        # of sufficient length. For lists shorter than this, it is probably
        # faster to just iterate than to go through the entire hashing process.
        if len(self.internal_methods_) > 10:
            self.internal_methods_ = set(self.internal_methods_)

        # Do NOT call this until after |internal_methods_| is set, or it will
        # lead to a crash, because the __getattribute__() call relies on it.
        super().__init__(*args, **kwargs)

    def __getattribute__(self, name):
        """
        Any call made to a method named in |internal_methods_| will be modified
        to allow calls to potentially equivariance-breaking PyTorch functions.
        This IS safe to use recursively.
        """
        attr = super().__getattribute__(name)
        if hasattr(attr, '__call__') and name in self.internal_methods_:
            def newCall(*args, **kwargs):
                # Get all |InternalType| instances.
                tracked_types : List[InternalType] = []
                for i, arg in enumerate(args):
                    if isinstance(arg, InternalType):
                        tracked_types.append(arg)
                        
                for _, value in kwargs.items():
                    if isinstance(value, InternalType):
                        tracked_types.append(value)

                # Allow internal calls.
                for instance in tracked_types:
                    instance.setInternal()

                # Make the original call.
                result = attr(*args, **kwargs)

                # Disallow internal calls.
                for instance in tracked_types:
                    instance.unsetState()

                return result
            return newCall
        else:
            return attr