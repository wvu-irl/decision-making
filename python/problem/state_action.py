from abc import ABC,abstractmethod

import numpy as np

class State():
    """
    Description: A class for representing states in MDPs. This includes sub states, possible actions, the value, policy, and whether or not a state is terminal, and hash function
    User defines:
        _state (list(int)): dictionary consisting of substates
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
        self.hash_ = None

        self.N_ = 0

        if type(_state) is list:
            self.s_ = _state
        else:
            self.s_ = [_state]
            
        if type(_action) is list:
            self.a_ = []
            for a in _action:
                self.a_.append(Action(a))
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
        return State(self.s_, self.a_, self.parent_, self.V_, self.is_terminal_, self.policy_)
    
    def __hash__(self):
        """
        Maps state values to hash key

        Returns:
            string: hash key
        """
        if self.hash_ == None:
            self.hash_ = hash(str(self.s_))

        return self.hash_
    
    def get_action_index(self, _a):
        for i in range(len(self.a_)):
            if self.a_[i].a_ == _a:
                return i
        return None
    
    def add_child(self, _a, _s_p, _s_p_i, _r):
        """
        Adds child action to state

        Args:
            _a (int): action id 
            _s_p (int): hash key for transition state
            _r (float): reward
        """
        ind = -1
        i = 0
        while ind == -1:
            if self.a_[i].a_ == _a:
                ind = i
            i +=1
        if ind == -1:
            self.a_.append(Action(_a))
            ind = len(self.a_)
        child_ind = self. a_[ind].add_child(_s_p, _s_p_i, _r)
        self.N_ += 1
        return child_ind
    
    # @abstractmethod 
    # def get_N(self):
    #     self.N_ = 0
    #     for a in _self.a_:
    #         self.N_ += a.N_
    
    def get_transition(self, _a):
        """
        Gets transition model associated with state-action

        Returns:
            list(tuple): transition model associated with state
        """
        for a in self.a_:
            if a.id == _a:
                return a.get_transition_model()

    def get_parent(self):
        """
        Gets parent states (in a graph, parents are considered as incoming nodes)

        Returns:
            list(string): gets hash key of parents
        """
        return self.parent_

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
    Description: A class for representing actions in MDPs. This includes resulting state hash keys and number of samples. Assumes reward is a weighted average of those received
    User defines:
        _action (int): id of action
    """
    def __init__(self, _action):
        """
        Constructor
        """
        super(Action, self).__init__()

        self.a_ = _action
        self.s_prime_ = []
        self.s_prime_i_ = []
        self.r_ = []
        self.n_ = []
        self.N_ = 0
    
    def add_child(self, _s, _s_p_i, _r):
        """
        Adds child state to action

        Args: print(self.tree_[nextNodeIndex].V_)
            _s_p (int): hash key for transition state
            _r (float): reward
        """
        ind = -1
        if _s in self.s_prime_:
            for x in range(len(self.s_prime_)): 
                if _s == self.s_prime_[x]:
                    i = x
                    break
            self.r_[i] = (self.n_[i]*self.r_[i] + _r)/(self.n_[i]+1)
            self.n_[i] += 1
            ind = i
        else:
            self.s_prime_.append(_s) 
            self.s_prime_i_.append(_s_p_i)
            self.r_.append(_r)
            self.n_.append(1)
            ind = _s_p_i
        self.N_ += 1
        return ind
    
    def get_transition_model(self):
        """
        Gets transition model associated with action

        Returns:
            list(string): has keys for children
            list(float): transition probabilities
            list(float): rewards
        """
        T = self.n_ / self.N_
        return self.s_prime_, T, self.r_
    
    def sample_transition_model(self, _rng):
        """
        Gets transition model associated with action

        Returns:
            list(string): has keys for children
            list(float): transition probabilities
            list(float): rewards
        """
        # print(self.N_)
        # print(self.n_)    
        p = _rng.random()*self.N_
        total = 0
        ind = 0
        while total < p:
            total += self.n_[ind]
            ind += 1
            
        return self.s_prime_[ind-1], self.r_[ind-1]
    
    def model_confidence(self):
        """
        Computes model confidence based on Hoeffding's Inequality

        Returns:
            float: confidence in model (0-1)
        """
        
        return 
        

    