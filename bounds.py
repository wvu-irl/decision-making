from abc import ABC,abstractmethod

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


class Bounds(ABC):
    def __init__(self):
        super(Bounds, self).__init__()

    @abstractmethod
    def get_lower(self): 
        pass
    
    @abstractmethod
    def get_upper(self):
        pass

class RmaxBound(Bounds):
    def __init__(self, _r_min = 0, _r_max = 1, _gamma = 0.95):
        super(Bounds, self).__init__()
        
        self.l_ = _r_min/(1-_gamma)
        self.u_ = _r_max/(1-_gamma)
        self.g_ = _gamma

    @abstractmethod
    def get_lower(self): 
        return self.l_
    
    @abstractmethod
    def get_upper(self):
        return self.u_