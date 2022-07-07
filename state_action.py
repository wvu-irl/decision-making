from abc import ABC,abstractmethod

import numpy as np

class State():
    """
    Description: A class for representing states in MDPs. This includes sub states, possible actions, the value, policy, and whether or not a state is terminal, and hash function
    User defines:
        _state (dict): dictionary consisting of substates
        _action (list(Action)): list of actions to take
        _parent (string): Hash key for parent
        _V (float): Initial estimate of the value
        _is_terminal (bool): is true if state is terminal
        _policy (int): Action to select
    """
    def __init__(self, _state, _action = None, _parent = None, _V = 0, _is_terminal = False, _policy = None):
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

        if type(_parent) is list:
            self.parent_ = _parent
        else:
            self.parent_ = []

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
    
    def add_child(self, _a, _s_p, _r):
        """
        Adds child action to state

        Args:
            _a (int): action id 
            _s_p (int): hash key for transition state
            _r (float): reward
        """
        
        pass
    
    def get_transition(self, _a):
        """
        Gets transition model associated with state-action

        Returns:
            list(tuple): transition model associated with state
        """
        
        pass

    def get_parent(self):
        """
        Gets parent states (in a graph, parents are considered as incoming nodes)

        Returns:
            list(string): gets hash key of parents
        """
        
        pass

    @abstractmethod
    def __eq__(self, _s):
        """
        Checks if two states are equal

        Args:
            _s (State): state to compare against

        Returns:
            bool: true if equal
        """
        return (self.s_ == _s.s_)
    
class Action():
    """
    Description: A class for representing actions in MDPs. This includes resulting state hash keys and number of samples
    User defines:
        _action (int): id of action
        _alpha (float): learning rate for reward
        _gamma (float): temporal discount factor
    """
    def __init__(self, _action, _alpha = 0.9, _gamma):
        """
        Constructor
        """
        super(Action, self).__init__()

        self.a_ = _action
        self.gamma_ = _gamma
        self.s_prime_ = []
        self.r_ = []
        self.n_ = []
        self.N_ = 0
    
    def add_child(self, _s, _r):
        """
        Adds child state to action

        Args: 
            _s_p (int): hash key for transition state
            _r (float): reward
        """
        # if state already added
            #increment n and add count to r
        # else 
            # add s, r, n to list
        self.N_ += 1
    
    def get_transition_model(self):
        """
        Gets transition model associated with action

        Returns:
            list(tuple): transition model associated with state
        """
        dist = []
        # get values, compute r + gamma*V
        t = self.n_ / self.N_
        #generate pairs and add to dist
        return dist
        #assume upper and lower bound/DST will be done by optimizer

    