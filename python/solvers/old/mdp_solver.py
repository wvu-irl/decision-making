from abc import ABC,abstractmethod

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from optimization.optimizer import Optimizer
from solvers.mdp_solver import MDPSolver

class MDPSolver(ABC):
    def __init__(_self, _optimizer, _mdp):
        super(MDPSolver, _self).__init__()

        _self.optimizer_ = _optimizer
        _self.mdp_ = _mdp

    @abstractmethod
    def evaluate(_self, _s=None):
        
        pass
    
    @abstractmethod
    def get_policy(_self, _s=None):
        pass
