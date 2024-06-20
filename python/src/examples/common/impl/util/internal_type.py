from typing import List

class InternalType:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tracked_children_ : List[InternalType] = []
        self.internal_depth_ = 0

    def setInternal(self, is_internal : bool = True):
        if not is_internal:
            assert self.internal_depth_ == 0
            return 
        
        self.internal_depth_ += 1

    def unsetState(self):
        assert self.internal_depth_ > 0
        self.internal_depth_ -= 1

        if not self.can_access_internals():
            for child in self.tracked_children_:
                child.unsetState()
                assert not child.can_access_internals()
            self.tracked_children_ = []
      
    def can_access_internals(self):
        return self.internal_depth_ > 0

    def addChild(self, child : 'InternalType'):
        assert isinstance(child, InternalType)
        self.tracked_children_.append(child)
        child.setInternal(self.internal_depth_ > 0)