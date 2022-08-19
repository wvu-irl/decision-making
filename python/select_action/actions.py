import math
import numpy as np
import random
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from problem.state_action import State, Action
from select_action.ambiguity_toolbox import *


class action_selection():
    """
    Selects which actions to take through a function
    Description: Class that defines which action is selected. 
    """
    def __init__(self, _func, _const : list = []):# -> None:
        """
        Constructor
        Args:
            Self (action_selection): action selection object
            _func (funciton): function for actions to be selected
            _const (list): List of constants to use for function
        Returns:
            actions_selection: Action Selection Object
        """
        self.func_ : function = _func
        self.const_ : list = _const
    
    def return_action(self,_s,_param = [], _solver = None):
        return self.func_(_s, self.const_,_param)


def UCB1(_s : State,_const,_param=[], _solver = None):
    UCB = math.nan
    optAction = math.nan
    for a in _s.a_:
        if not(a.N_ == 0 or _s.N_ == 0):
            aVal = _s.V_ + _const["c"]*np.sqrt(((2*np.log(a.N_))/_s.N_)) #UCB1 Equation
        else:
            aVal = _s.V_
        if aVal > UCB or np.isnan(UCB):
            UCB = aVal
            optAction = a.a_
    return optAction

#assume that when initialized it gets alpha, but the
#solver can override, for example when doing optimis    random.shuffle(_s.a_)tic search
def ambiguity_aware(_s,_const = 1,_params=[], _solver = None):
    epsilon = _solver.performance_[0]
    delta = _solver.performance_[1]
    gamma = _solver.gamma
    L = _solver.bounds_[0]
    U = _solver.bounds_[1]
    if _params == []:
        alpha = _const[0]
    else:
        alpha = _params[0]
    
    max_expectation = -inf
    ind = 0
    gap = 0
    
    for a in _s.a_:
        dist, t = count_2_dist(a, gamma, _solver)
        # dist -> distribution (a, r+gamma V)
        # t -> number of samples
        
        bf = dist_2_bf(dist, t, epsilon, L, U)
        
        low_exp = lower_expectation(bf)
        up_exp = upper_expectation(bf)
        expectation = (1-alpha)*low_exp + (alpha)*up_exp #+ 0.5**np.sqrt(np.log(N)/t)
        if expectation > exp_max:
            exp_max = expectation
            gap = up_exp-low_exp
            ind = a.a_
    return ind, exp_max, gap


def randomAction(_s : State,_const,_param,solver = None):
    return np.random.randint(len(_s.a_))
    