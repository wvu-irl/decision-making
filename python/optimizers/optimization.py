from abc import ABC,abstractmethod

import numpy as np
from numpy import argmax

class Optimizer(ABC):
    def __init__(_self, _problem, *_params):
        super(Optimizer, _self).__init__()
        _self.problem_ = _problem
            
    @abstractmethod
    def compute_Q(_self, _s):
        pass

    def get_unique(self, _lst):
        lst = []
        for item in _lst:
            if item not in lst:
                lst.append(item)
        return lst
    
class Bellman(Optimizer):
    def __init__(_self, _problem, *_params):
        super().__init__(_problem, _params)
            
    def compute_Q(_self, _s):
        #print("Compute Q")
        spm, t = _self.problem_.get_transition_model( _s, _a, False)
        # print(t)
        # print("--------")
        Q = 0
        for i in range(len(spm)):
            sp = np.ravel_multi_index(spm[i], _self.problem_.get_num_states())
            Q += t[i]* (_self.problem_.get_reward(_s,_a, spm[i]) +_gamma*np.max(_Q[sp]))
        return Q

    def compute_V(_self, _s):
        pass
    
class PignisticBellman(Optimizer):
    def __init__(_self, _problem, *_params):
        super().__init__(_problem, _params)
            
    def compute_Q(_self, _s):
        
        spm, t = _self.problem_.get_transition_model( _s, _a, True)
        
        Q = 0
        for i in range(len(spm)):
            sp = np.ravel_multi_index(spm[i], _self.problem_.get_num_states())
            Q += t[i]* (_self.problem_.get_reward(_s,_a, spm[i]) +_gamma*np.max(_Q[sp]))

        return Q

# class Bellman(Optimizer):
#     def __init__(_self, _problem, *_params):
#         super().__init__(_problem, _params)
            
#     def compute_Q(_self, _s):
#         #print("Compute Q")
#         spm, t = _self.problem_.get_transition_model( _s, _a, False)
#         # print(t)
#         # print("--------")
#         Q = 0
#         for i in range(len(spm)):
#             sp = np.ravel_multi_index(spm[i], _self.problem_.get_num_states())
#             Q += t[i]* (_self.problem_.get_reward(_s,_a, spm[i]) +_gamma*np.max(_Q[sp]))
#         return Q
    
# class PignisticBellman(Optimizer):
#     def __init__(_self, _problem, *_params):
#         super().__init__(_problem, _params)
            
#     def compute_Q(_self, _s, _a, _gamma, _Q):
        
#         spm, t = _self.problem_.get_transition_model( _s, _a, True)
        
#         Q = 0
#         for i in range(len(spm)):
#             sp = np.ravel_multi_index(spm[i], _self.problem_.get_num_states())
#             Q += t[i]* (_self.problem_.get_reward(_s,_a, spm[i]) +_gamma*np.max(_Q[sp]))

#         return Q

# #######################################################################################
# class Bellman(Optimizer):
#     def __init__(_self, _problem, *_params):
#         super().__init__(_problem, _params)
            
#     def compute_Q(_self, _s, _a, _gamma, _Q):
#         #print("Compute Q")
#         spm, t = _self.problem_.get_transition_model( _s, _a, False)
#         # print(t)
#         # print("--------")
#         Q = 0
#         for i in range(len(spm)):
#             sp = np.ravel_multi_index(spm[i], _self.problem_.get_num_states())
#             Q += t[i]* (_self.problem_.get_reward(_s,_a, spm[i]) +_gamma*np.max(_Q[sp]))
#         return Q
    
# class PignisticBellman(Optimizer):
#     def __init__(_self, _problem, *_params):
#         super().__init__(_problem, _params)
            
#     def compute_Q(_self, _s, _a, _gamma, _Q):
        
#         spm, t = _self.problem_.get_transition_model( _s, _a, True)
        
#         Q = 0
#         for i in range(len(spm)):
#             sp = np.ravel_multi_index(spm[i], _self.problem_.get_num_states())
#             Q += t[i]* (_self.problem_.get_reward(_s,_a, spm[i]) +_gamma*np.max(_Q[sp]))

#         return Q


# class Bellman(Optimizer):
#     def __init__(_self, _problem, *_params):
#         super(Optimizer, _self).__init(_problem, _params)
            
#     @abstractmethod
#     def get_value(_self, _s, _actions, _s_prime):
#         unique_a = []
#         V = np.zeros([len(unique_a),1])
#         for i in range(len(_actions)):
#             a_idx = [a for a in unique_a if _actions[i] == a]
#             T = _self.problem_.transition_prob(_s,_actions[i], _s_prime[i])
#             V[a_idx] += T* (_self.problem_.reward(_s,_actions[i], _s_prime[i]) + _self.problem_.gamma*_s_prime[i].V)

#         return max(V), unique_a(argmax(V))
    

# class PignisticBellman(Optimizer):
#     def __init__(_self, _problem, *_params):
#         super(Optimizer, _self).__init(_problem, _params)
            
#     @abstractmethod
#     def optimize(_self, _s, _actions, _s_prime):
#         unique_a = []
#         V = np.zeros([len(unique_a),1])
#         for i in range(len(_actions)):
#             a_idx = [a for a in unique_a if _actions[i] == a]
#             T = _self.problem_.transition_prob(_s,_actions[i], _s_prime[i])
#             V[a_idx] += T* (_self.problem_.reward(_s,_actions[i], _s_prime[i]) + _self.problem_.gamma*_s_prime[i].V)

#         return max(V)