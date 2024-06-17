from abc import abstractmethod

class TensorRecurserClient:
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def _getParts(self):
        raise NotImplementedError("This method must be implemented!")
    
    @abstractmethod
    def _createObject(self, vals):
        raise NotImplementedError("This method must be implemented!")