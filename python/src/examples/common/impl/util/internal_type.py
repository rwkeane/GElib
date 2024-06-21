from typing import List

class InternalType:
    """
    Tracks when a class should be allow to access "internal only" properties. To
    be used with InternalCaller.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tracked_children_ : List[InternalType] = []
        self.internal_depth_ = 0
      
    def can_access_internals(self):
        return self.internal_depth_ > 0

    def set_depth(self, depth : int):
        assert depth >= 0
        self.internal_depth_ = depth

    def depth(self):
        return self.internal_depth_

    def setInternal(self):
        self.internal_depth_ += 1

    def unsetState(self):
        assert self.internal_depth_ > 0, self
        self.internal_depth_ -= 1

        if not self.can_access_internals():
            for child in self.tracked_children_:
                child.unsetState()
                assert not child.can_access_internals()
            self.tracked_children_ = []

    def addChild(self, child : 'InternalType'):
        assert isinstance(child, InternalType)
        if self.internal_depth_ <= 0:
            return
        
        self.tracked_children_.append(child)
        child.setInternal()