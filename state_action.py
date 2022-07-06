from abc import ABC,abstractmethod

import numpy as np

class State():
    """
    Description: A class for representing states in MDPs. This includes sub states, possible actions, the value, policy, and whether or not a state is terminal, and hash function
    User defines:
        _state (dict): dictionary consisting of substates
        _action (list(Action)): list of actions to take
        _V (float): Initial estimate of the value
        _is_terminal (bool): is true if state is terminal
        _policy (int): Action to select
    """
    def __init__(self, _state, _action = None, _V = 0, _is_terminal = False, _policy = None):
        """
        Constructor
        """
        super(State, self).__init__()

        self.V_ = _V
        self.policy_ = _policy
        self.is_terminal_ = _is_terminal

        self.s_ = _state

        if type(_state) is dict:
            self.s_ = _state
        else:
            self.s_ = {'x': _state}
            
        if type(_action) is list:
            self.a_ = _action
        else:
            self.a_ = []

    def __copy__(self):
        """
        copy function

        Returns:
            State: copied state
        """
        return State(self.s_, self.V_, self.policy_, self.is_terminal_)
    
    def __hash__(self):
        """
        Maps state values to hash key

        Returns:
            string: hash key
        """
        pass
    
    def add_child(self, _a):
        """
        Adds child action to state

        Args:
            _a (Action): 
        """
        
        pass
    
    def get_transition(self, _a):
        """
        Gets transition model associated with state

        Returns:
            list(tuple): transition model associated with state
        """
        
        pass

    @abstractmethod
    def __eq__(self, _s):
        """
        Constructor, initializes MDP model

        Args:
            self (VI_Adversary): Object to initialize
            _epsilon (double): convergence criteria
            _gamma (double): Discount factor
            _num_actions (int): number of actions

        Returns:
            VI_Adversary: Q-value iteration object
        """
        return (self.s_ == _s.s_)
    
class Action():
    def __init__(self, _action, _gamma):
        super(Action, self).__init__()

        self.a_ = _action
        self.gamma_ = _gamma
        self.s_prime_ = []
        self.r_ = []
        self.n_ = []
        self.N_ = 0
    
    def add_child(self, _s):
        # if state already added
            #increment n and add count to r
        # else 
            # add s, r, n to list
        self.N_ += 1
    
    def get_distribution(self):
        dist = []
        # get values, compute r + gamma*V
        t = self.n_ / self.N_
        #generate pairs and add to dist
        return dist
        #assume upper and lower bound/DST will be done by optimizer


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