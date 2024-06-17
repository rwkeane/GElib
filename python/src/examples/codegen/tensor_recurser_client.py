from abc import abstractmethod

class TensorRecurserClient:
    def __init__(self):
        pass
    
    @abstractmethod
    def __getParts(self):
        raise NotImplementedError("This method must be implemented!")
    
    @abstractmethod
    def __createObject(self, vals):
        raise NotImplementedError("This method must be implemented!")