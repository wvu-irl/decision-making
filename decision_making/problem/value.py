from abc import ABC,abstractmethod

import numpy as np

class Value():
    """
    Description: A class for representing states in MDPs. This includes sub states, possible actions, the value, policy, and whether or not a state is terminal, and hash function
    User defines:
        _r (list(int)): value observations
    """
    def __init__(self, _r):
        """
        Constructor
        """
        super(Value, self).__init__()

        if type(_r) is list:
            self.r_ = _r
        else:
            self.r_ = []
            
        self.R_ = self.compute_reward()

    def __copy__(self):
        """
        copy function

        Returns:
            State: copied state
        """
        return Value(self.r_)
    
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

    def compute_reward(self):
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

    @abstractmethod
    def __gt__(_a):
        pass    

    @abstractmethod
    def __lt__(_a):
        pass    

    @abstractmethod
    def __ge__(_a):
        pass    

    @abstractmethod
    def __le__(_a):
        pass    
  
    @abstractmethod
    def __ne__(_a):
        pass   

    ## Math ---------------------------------------------------

    @abstractmethod
    def __add__(_a):
        pass 
    
    @abstractmethod
    def __sub__(_a):
        pass   

    @abstractmethod
    def __mul__(_a):
        pass 

    @abstractmethod
    def __truediv__(_a):
        pass 

    @abstractmethod
    def __iadd__(_a):
        pass 
    
    @abstractmethod
    def __isub__(_a):
        pass   

    @abstractmethod
    def __imul__(_a):
        pass 

    @abstractmethod
    def __idiv__(_a):
        pass 

    @abstractmethod
    def __pow__(_a):
        pass  