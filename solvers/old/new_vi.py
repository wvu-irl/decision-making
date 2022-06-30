import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from optimization.optimizer import Optimizer
from solvers.mdp_solver import MDPSolver

class VI(MDPSolver):
    def __init__(_self, _optimization, _epsilon = 1):
        super(MDPSolver, _self).__init__(_optimization)

        _self.epsilon_ = _epsilon
        _self.states_ = _self.optimization_.mdp_.ss_.get_states()

    def evaluate(_self):
        V_max = -np.inf
        epsilon = 5*_self.epsilon_
        while epsilon > _self.epsilon_:
            for s in _self.states_:
                for a,sp in _self.optimization_.mdp_.get_transitions(s):
                    v_prev = s.V
                    s.V_, s.policy_ = _self.optimizer_.get_value(s, a, sp)
                    epsilon = np.max(np.fabs(s.V - v_prev), epsilon)

    
    def get_policy(_self, _s=None):
        return _s.policy_