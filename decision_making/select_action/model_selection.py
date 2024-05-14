import math
import numpy as np
import sys
import os
import copy

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from problem.state_action import State

def sample_ambiguous_dist(dist, p):
    """
    Description: Samples a distribution with ambiguity
    
    :param dist: Distribution to sample
    :param p: probability of threshold
    :return: Sample
    """
    total = 0
    i = 0
    while total <= p:
        total += dist[i]["p"]
        i += 1
    if len(dist[i-1]["el"]) == 1:
        return dist[i-1]["el"][0]
    else:
        return np.random.choice(dist[i-1]["el"])

class model_selection():
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
    
    def return_model(self,_s,_param = {}, _solver = None):
        return self.func_(_s, self.const_,_param,_solver)
    
def mm_progressive_widening(_s : State,_const,_param={},_solver = None):
    """
    Description: Progressive Widening function for selecting model
    
    :param _s: State object
    :param _const: Constants for function
    :param _param: Parameters for function
    :return: Model to select
    """
    if _param == {}:
        k = _const["k"]
        a = _const["a"]
    else:
        k = _param["k"]
        a = _param["a"]
        
    # raise NotImplementedError("Need to fix sampling to make sure model id is conveyed properly")
        
    if len(_s.m_) <= k*_s.N_**a and len(_s.m_):

        models = []
        for m in _s.model_dist_.keys():
            if type(m) is set:
                els = []
                for el in m:
                    if el in _s.m_:
                        els.append(m)
                models.append({"el":els, "p":_s.model_dist_[m]})
            else:
                if m in _s.m_:
                    models.append({"el":[m], "p":_s.model_dist_[m]})
            
                    
        total = 0
        for m in models:
            total += m["p"]
        for i in range(len(models)):
            models[i]["p"] = models[i]["p"]/total
        
        p = np.random.rand()
        print(p)
        return sample_ambiguous_dist(models, p)
    
    
    else:
        unused_models = []
        for m in _s.model_dist_.keys():
            if type(m) is set:
                els = []
                for el in m:
                    if el in _s.m_unused_:
                        els.append(m)
                unused_models.append({"el":els, "p":_s.model_dist_[m]})
            else:
                if m in _s.m_unused_:
                    unused_models.append({"el":[m], "p":_s.model_dist_[m]})
            
        total = 0
        for m in unused_models:
            total += m["p"]
        for i in range(len(unused_models)):
            unused_models[i]["p"] = unused_models[i]["p"]/total
            
        p = np.random.rand()
        return sample_ambiguous_dist(unused_models, p)
    