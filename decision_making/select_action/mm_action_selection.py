import math
from kiwisolver import Solver
import numpy as np
import random
import sys
import os
from copy import deepcopy

from pandas import option_context

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from problem.mm_state_action import State, Action
from select_action.ambiguity_toolbox import *
from select_action.gbop_toolbox import *

from select_action.model_selection import model_selection
from select_action.utils import * 

class mm_action_selection():
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
        # print("umm")
        self.const_["model_sel_func"] = model_selection(act_sel_funcs[_const["model_selection"]["function"]], _const["model_selection"]["params"])
    
    def return_action(self,_s,_param = {}, _solver = None):
        return self.func_(_s, self.const_,_param,_solver)

def ambiguity_aware(_s,_const = 1,_params={}, _solver = None):
        # if _params[1] == None:
        #     no_c = True
    # print(alpha)
    exp_max = -inf
    best_model = 0
    best_action = None #_s.a_[best_action].a_
    gap = 0
    lexps = []
    uexps = []
    if "alpha" not in _params:
        _params["alpha"] = _const["alpha"]
    # if _s.s_ == {"pose": [17,16]}:
    #     print(_s)
    # print(";;;;;;;;;;;;;;;;;;")
    # print(_s.s_)
    print_a_values = []
    
    if "is_policy" in _params and _params["is_policy"]:
        for model in _s.m_:
            for a in _s.a_[model]:

                expectation, L_exp, U_exp = get_action_expectation(a, _const, _params, _solver)
                
                best_model, best_action, exp_max, gap, lexps, uexps = update_best_action(best_model, best_action, model, a, expectation, L_exp, U_exp, exp_max, gap, lexps, uexps)
                
                print_a_values.append((model, a.a_, expectation, L_exp, U_exp))
                
    else:
        
        # mm_params = _const["model_selection"]["params"]
        
        model = _solver.model_sel_.return_model(_s, {}, _solver)#_const["model"], mm_params, _solver)
        a_params = _const["action_prog_widening"]
        a_params["m"] = model
        
        is_widened, a = progressive_widening(_s, _const, a_params, _solver)
        
        if not is_widened:
            # print("----NOT WIDENED----")
            
            for a in _s.a_[model]:
                expectation, L_exp, U_exp = get_action_expectation(a, _const, _params, _solver)
                
                best_model, best_action, exp_max, gap, lexps, uexps = update_best_action(best_model, best_action, model, a, expectation, L_exp, U_exp, exp_max, gap, lexps, uexps)  
                print_a_values.append((a.a_, expectation, L_exp, U_exp))
                
        else:
            a = Action(a)
            expectation, L_exp, U_exp = get_action_expectation(a, _const, _params, _solver)
            best_model, best_action, exp_max, gap, lexps, uexps = update_best_action(best_model, best_action, model, a, expectation, L_exp, U_exp, exp_max, gap, lexps, uexps)  

    # print(L_exp, U_exp)
    ldiff = 0
    udiff = 0
    if len(uexps) > 1:
        uexps.sort()
        lexps.sort()    
        ldiff = lexps[0]-lexps[1]
        udiff = uexps[0]-uexps[1]
    
    ldiff = max(0.1,ldiff)
    udiff = max(0.1,udiff)
    # print(lexps,uexps)
    # raise("print out expectations and actions")
    # raise("check termination reward")

    a = {"model": best_model, **best_action}
    return a, exp_max, L_exp, U_exp, [ldiff, udiff], [lexps,uexps], print_a_values

def random_action(_s : State,_const,_param,solver = None):
    a = []
    for a_t in _s.a_:
        a.append(a_t.a_)
    # print(a)
    return np.random.choice(a)
    
def progressive_widening(_s : State, _const, _param, solver = None):
    if _param == {}:
        k = _const["k"]
        a = _const["a"]
    else:
        k = _param["k"]
        a = _param["a"]
    
    if len(_s.a_[_param["m"]]) == 0:
        return True, solver.env_.get_action(_param["m"], _s.s_)
    # print("mm", len(_s.a_[_param["m"]]), _s.N_, _s.model_N_[_param["m"]], k*_s.model_N_[_param["m"]]**a, len(_s.a_[_param["m"]]) <= k*_s.model_N_[_param["m"]]**a)
    if len(_s.a_[_param["m"]]) <= k*_s.model_N_[_param["m"]]**a:
        a = None
        timeout = 0
        while a == None and timeout < 20:
            temp_a = solver.env_.get_action(_param["m"], _s.s_)
            select_a = True
            for act in _s.a_[_param["m"]]:
                if act.a_ == temp_a:
                    select_a = False
            if select_a:
                a = temp_a
                # _s.add_child(a)
            timeout += 1
        if timeout == 20:
            # a = np.random.choice(_s.a_[_param["m"]]).a_
            return False, None
        return True, a
        
    else:
        # a = np.random.choice(_s.a_[_param["m"]])
        return False, None

def get_action_expectation(_a,_const = 1,_params={}, _solver = None):
    delta = 1-_solver.alg_params_["model_accuracy"]["delta"]
    epsilon = _solver.alg_params_["model_accuracy"]["epsilon"]
    L = _solver.bounds_[0]*(_solver.search_params_["horizon"]-_solver.d_-1)
    U = _solver.bounds_[1]*(_solver.search_params_["horizon"]-_solver.d_-1)
    gamma = _solver.alg_params_["gamma"]
    no_c = False
    if _params == []:
        alpha = _const["alpha"]
    else:
        alpha = _params["alpha"]
         
    if _a.N_ == 0:
        expectation = (1-alpha)*L + (alpha)*U
        low_exp = L
        up_exp = U
    else:
        dist, t = count_2_dist(_a, gamma, _solver, True)

        bf = generate_bf_conf(dist, delta, t, L, U, epsilon)
        up_exp = upper_expectation(bf)
        dist, t = count_2_dist(_a, gamma, _solver, False)
        bf = generate_bf_conf(dist, delta, t, L, U, epsilon)
        # print("bfl", bf)
        low_exp = lower_expectation(bf)

        expectation = (1-alpha)*low_exp + (alpha)*up_exp #+ 0.5**np.sqrt(np.log(N)/t)

    return expectation, low_exp, up_exp

def update_best_action(best_model, best_action, model, a, expectation, low_exp, up_exp, exp_max, gap, lexps, uexps):
    
    if type(best_model) is not list:
        best_model = [best_model]
    if type(best_action) is not list:
        best_action = [best_action]
    
    uexps.append(up_exp)
    lexps.append(low_exp)
    if expectation > exp_max:
        exp_max = expectation
        # L_exp = low_exp
        # U_exp = up_exp
        gap = up_exp-low_exp
        best_model = [model]
        best_action = [a.a_]
    
    elif expectation == exp_max:
        best_model.append(model)
        best_action.append(a.a_)
    
    i = np.random.choice(range(len(best_model)))
    best_model = best_model[i]
    best_action = best_action[i]
    
    return best_model, best_action, exp_max, gap, lexps, uexps