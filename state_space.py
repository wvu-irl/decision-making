from abc import ABC,abstractmethod

import numpy as np

class State():
    def __init__(self, _state, _V = 0, _policy = None, _is_terminal = False):
        super(State, self).__init__()

        self.V_ = _V
        self.policy_ = _policy
        self.is_terminal_ = _is_terminal

        self.s_ = _state
        
        #how to structure children? Need to have actions, then actions carry

        if type(_state) is dict:
            self.s_ = _state
        else:
            self.s_ = {'x': _state}

    def __copy__(self):
        return State(self.s_, self.V_, self.policy_, self.is_terminal_)
    
    def __hash__(self):
        pass
    
    def add_child(self, _a, _s):
        pass

    @abstractmethod
    def __eq__(self, _s):
        return (self.s_ == _s.s_)
    
class Action():
    def __init__(self, _action):
        super(Action, self).__init__()

        self.a_ = _action
        self.s_prime_ = []
        self.r_ = []
        self.n_ = []
    
    def add_child(self, _s):
        # if state already added
            #increment n and add count to r
        # else 
            # add s, r, n to list
        pass
    
    def get_distribution(self):
        dist = []
        return dist

    @abstractmethod
    def __eq__(self, _s):
        return (self.s_ == _s.s_)
    
# class StateActionState():
#     def __init__(self, _state, _V = 0, _policy = None, _is_terminal = False):
#         super(State, self).__init__()

#         self.V_ = _V
#         self.policy_ = _policy
#         self.is_terminal_ = _is_terminal

#         self.s_ = _state

#         if type(_state) is dict:
#             self.s_ = _state
#         else:
#             self.s_ = {'x': _state}

#     def __copy__(self):
#         return State(self.s_, self.V_, self.policy_, self.is_terminal_)
    
#     def __hash__(self):
#         pass

#     @abstractmethod
#     def __eq__(self, _s):
#         return (self.s_ == _s.s_)    


class StateSpace():
    def __init__(self, _states: State):
        super(StateSpace, self).__init__()

        self.S_ = _states
        self.N_ = len(self.S_)

        self.rng_ = np.random.default_rng()
        

    def is_well_formed_state(self, _s):
        return (self.S_.keys() == _s.keys())


    @abstractmethod
    def is_state(self, _s):
        if self.is_well_formed_state(_s) and _s in self.S_:
            return True
        return False

    @abstractmethod
    def get_random_state(self):
        return self.state_(self.rng_.choice(self.N_))
        
    @abstractmethod
    def get_random_state(self):
        pass

    @abstractmethod
    def enumerate_states(self):
        pass

    @abstractmethod
    def state_2_ind(self, _s):
        pass

    @abstractmethod
    def ind_2_state(self, _ind):
        pass

    @abstractmethod
    def is_terminal(self, _s):
        pass