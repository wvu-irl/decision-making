import math
from kiwisolver import Solver
import numpy as np
import random
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from problem.state_action import State, Action
from select_action.ambiguity_toolbox import *
from select_action.gbop_toolbox import *


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
        return self.func_(_s, self.const_,_param,_solver)


def UCB1(_s : State,_const,_param=[],_solver = None):
    UCB = math.nan
    optAction = math.nan
    actions = _s.a_ 
    random.shuffle(actions)
    for a in actions:
        if len(a.s_prime_i_) > 0:
            childInd = a.s_prime_i_[0]
            aVal = _solver.tree_[childInd].V_/_solver.tree_[childInd].N_ + 2*_const["c"]*np.sqrt(((np.log(_s.N_))/_solver.tree_[childInd].N_)) #UCB1 Equation
            if aVal > UCB or np.isnan(UCB):
                UCB = aVal
                optAction = a.a_
    return optAction

def ambiguity_aware(_s,_const = 1,_params=[], _solver = None):
    pass


#assume that when initialized it gets alpha, but the
#solver can override, for example when doing optimis    random.shuffle(_s.a_)tic search
def ambiguity_aware(_s,_const = 1,_params=[], _solver = None):
    epsilon = _solver.performance_[0]
    delta = _solver.performance_[1]
    gamma = _solver.gamma_
    L = _solver.bounds_[0]
    U = _solver.bounds_[1]
    no_c = False
    if _params == []:
        alpha = _const[0]
    else:
        alpha = _params[0]
        # if _params[1] == None:
        #     no_c = True
    
    exp_max = -inf
    exp_max2 = -inf
    ind = 0
    gap = 0
    lexps = []
    uexps = []
    for a in _s.a_:
        if a.N_ == 0:
            expectation = (1-alpha)*L + (alpha)*U
            low_exp = L
            up_exp = U
        else:
            
            dist, t = count_2_dist(a, gamma, _solver, True)
            # dist -> distribution (a, r+gamma V)
            # t -> number of samples
            bf = dist_2_bf(dist, t, epsilon, L, U, no_c)
            up_exp = upper_expectation(bf)
            # print(bf)
            
            dist, t = count_2_dist(a, gamma, _solver, False)
            bf = dist_2_bf(dist, t, epsilon, L, U, no_c)
            low_exp = lower_expectation(bf)
            # print(low_exp)
            
            # print(up_exp)
            expectation = (1-alpha)*low_exp + (alpha)*up_exp #+ 0.5**np.sqrt(np.log(N)/t)
            # print(alpha)
            # print("exp", expectation)
            #exps.append(expectation)
            uexps.append(up_exp)
            lexps.append(low_exp)
        if expectation > exp_max:
            exp_max = expectation
            L_exp = low_exp
            U_exp = up_exp
            gap = up_exp-low_exp
            ind = [a.a_]
            
        elif expectation == exp_max:
            ind.append(a.a_)
        ldiff = 0
        udiff = 0
        if len(uexps) > 1:
            uexps.sort()
            lexps.sort()    
            ldiff = lexps[0]-lexps[1]
            udiff = uexps[0]-uexps[1]
        
        ldiff = max(0.1,ldiff)
        udiff = max(0.1,udiff)
    return _solver.rng_.choice(ind), exp_max, L_exp, U_exp, [ldiff, udiff]


def randomAction(_s : State,_const,_param,solver = None):
    return np.random.randint(len(_s.a_))
    