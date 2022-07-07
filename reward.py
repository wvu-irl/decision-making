from abc import ABC,abstractmethod

import numpy as np

class Reward():
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
    
    def update_reward(self, _r):
        """
        Maps state values to hash key

        Returns:
            string: hash key
        """
        #treat rewards like obeservations, 
            # take in params like update rules, need reward rules
        #returns reward 

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

        #overload operators