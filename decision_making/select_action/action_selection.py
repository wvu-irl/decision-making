import math
from kiwisolver import Solver
import numpy as np
import random
import sys
import os
import copy

from pandas import option_context

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
    def __init__(self, _func, _const : dict = {}):# -> None:
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
        self.const_ : dict = _const
    
    def return_action(self,_s,_param = [], _solver = None):
        return self.func_(_s, self.const_,_param,_solver)


def UCB1(_s : State,_const,_param=[],_solver = None):
    UCB = math.nan
    optAction = math.nan
    actions = copy.deepcopy(_s.a_ )
    random.shuffle(actions)
    if _param == []:
        c = _const["c"] 
    else:
        c = _param[0]
    for a in actions:
        Q = 0
        if a.N_ != 0:
            for r,s_p_i,n in zip(a.r_, a.s_prime_i_, a.n_):
                Q += n*(r + _solver.alg_params_["gamma"]*_solver.tree_[s_p_i].V_)#/_solver.tree_[s_p_i].N_  #UCB1 Equation
        # if a.N_ == 0:
            # Q = np.inf
            Q /= a.N_
            Q += 2*c*np.sqrt(((np.log(_s.N_))/a.N_))
        if Q > UCB or np.isnan(UCB):
                UCB = Q
                optAction = a.a_
    if np.isnan(optAction):
        optAction = actions[0].a_
    return optAction

def gbop_dm(_s,_const = 1,_params=[], _solver = None):
    L = _solver.bounds_[0]
    U = _solver.bounds_[1]
    
    U_max = -inf
    L_min = inf

    Us = []
    Ls = []

    for a in _s.a_:
        if a.N_ == 0:
            low_b = L
            up_b = U
        else:
            up_b = boundSolver(_s,_solver,"upper")
            low_b = boundSolver(_s,_solver,"lower")
        if up_b > U_max:
            U_max = up_b
        if low_b < L_min:
            L_min = low_b
            
        Us.append(up_b)
        Ls.append(low_b)
    # print(U_max)
    return L_min, U_max

def gbop_best_action(_s,_const = [],_params=[], _solver = None):
    bestAction = None
    bestValue = -np.inf
    actions = _s.a_ 
    random.shuffle(actions)
    if _params == []:
        alpha = _const["alpha"]
    else:
        alpha = _params[0]
    for a in actions:
        V = 0
        for s in a.s_prime_:
            s_p = _solver.graph_[_solver.gi_[hash(str(s))]]
            # if _params[0] == 1:
            #     V += sum(a.r_)/len(a.r_) + _solver.gamma_*s_p.U_
            #     # print("U",s_p.U_)
            # else:
            V += alpha * (sum(a.r_)/len(a.r_) + _solver.alg_params_["gamma"]*s_p.U_)
            V += (1-alpha) * (sum(a.r_)/len(a.r_) + _solver.alg_params_["gamma"]*s_p.L_)
                # print("L",s_p.L_)
        if len(a.s_prime_) == 0:
            if alpha == 1:
                V = _solver.bounds_[1]
            else:
                V = _solver.bounds_[0]

        if V > bestValue:
            bestAction = a
            bestValue = V
        if bestAction == None:
            bestAction = a
    return bestAction.a_

#assume that when initialized it gets alpha, but the
#solver can override, for example when doing optimis    random.shuffle(_s.a_)tic search
def ambiguity_aware(_s,_const = 1,_params=[], _solver = None):
    delta = 1-_solver.alg_params_["model_accuracy"]["delta"]
    epsilon = _solver.alg_params_["model_accuracy"]["epsilon"]
    L = _solver.bounds_[0]
    U = _solver.bounds_[1]
    gamma = _solver.alg_params_["gamma"]
    no_c = False
    if _params == []:
        alpha = _const["alpha"]
    else:
        alpha = _params[0]
        # if _params[1] == None:
        #     no_c = True
    # print(alpha)
    exp_max = -inf
    ind = _s.a_[0].a_
    gap = 0
    lexps = []
    uexps = []
    # if _s.s_ == {"pose": [17,16]}:
    #     print(_s)
    counta = 0
    # print(";;;;;;;;;;;;;;;;;;")
    # print(_s.s_)
    for a in _s.a_:
        if a.N_ == 0:
            expectation = (1-alpha)*L + (alpha)*U
            low_exp = L
            up_exp = U
            counta += 1
        else:
            # print("--")
            # print(a.N_)
            # print(a.n_)
            # print("s", _s.s_, a.a_)
            dist, t = count_2_dist(a, gamma, _solver, True)
            # dist -> distribution (a, r+gamma V)
            # t -> number of samples
            #bf = dist_2_bf(dist, t, epsilon, L, U, no_c)
            # print("------------")
            # print("dist", dist)
            bf = generate_bf_conf(dist, delta, t, L, U, epsilon)
            up_exp = upper_expectation(bf)
            # print(bf)
            # print(up_exp)
            # print("bfu", bf)
            dist, t = count_2_dist(a, gamma, _solver, False)
            # print(dist)
            #bf = dist_2_bf(dist, t, epsilon, L, U, no_c)
            bf = generate_bf_conf(dist, delta, t, L, U, epsilon)
            # print("bfl", bf)
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
        # if int(_s.s_[0]) == 20 and int(_s.s_[1]) == 20 :
            # print("------------")
            # print(U_exp)
            # print(exp_max)
            # print(L_exp)
    # if _s.s_ == {"pose": [20, 19]}:
    #     print(L_exp)
    #     print(U_exp)
    # if _s.s_ == {"pose": [17,16]}:
    # if _s.s_["pose"][0] <15 and _s.s_["pose"][1] <15 and alpha == 1:
    #     print(_s.s_, counta)
    #     print(lexps)
    #     print(uexps)
        
    return _solver.rng_.choice(ind), exp_max, L_exp, U_exp, [ldiff, udiff], [lexps,uexps]

def random_action(_s : State,_const,_param,solver = None):
    a = []
    for a_t in _s.a_:
        a.append(a_t.a_)
    # print(a)
    return np.random.choice(a)
    