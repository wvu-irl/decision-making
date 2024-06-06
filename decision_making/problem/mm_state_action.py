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
    def __init__(self, _state, _model_dist = None, _parent = None, _V = 0, _is_terminal = False, _policy_m = None, _policy_a = None, _L = None, _U = None):
        """
        Constructor
        """
        super(State, self).__init__()

        self.V_ = _V
        if _U == None:
            self.U_ = _V
        else:
            self.U_ = _U
        if _L == None:
            self.L_ = _V
        else:
            self.L_ = _L
        self.is_terminal_ = _is_terminal
        
        self.s_ = _state
        self.hash_ = None

        self.N_ = 0

        if type(_state) is dict:
            self.s_ = _state
        else:
            self.s_ = {"x":_state}
        
        self.m_ = []
        self.m_unused_ = []
        if type(_model_dist) is dict:
            for m in _model_dist.keys():
                if type(m) is set:
                    for el in m:
                        if el not in self.m_unused_:
                            self.m_unused_.append(el)
                else:
                    if m not in self.m_unused_:
                        self.m_unused_.append(m)
                        
            self.model_dist_ = _model_dist
            
        self.a_ = {}
        self.model_N_ = {}
        for m in self.m_unused_:
            self.a_[m] = []
            self.model_N_[m] = 0
        
        # if type(_action) is list:
        #     self.a_ = []
        #     self.a_unused = []
        #     for a in _action:
        #         self.a_.append(Action(a))
        #         self.a_unused.append(a)
        # else:
        #     self.a_ = []
            
        self.policy_m_ = _policy_m
        self.policy_a_ = _policy_a

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
        return State(self.s_, self.m_, self.parent_, self.V_, self.is_terminal_, self.policy_)
    
    def __hash__(self):
        """
        Maps state values to hash key

        Returns:
            string: hash key
        """
        if self.hash_ == None:
            self.hash_ = hash(str(self.s_))

        return self.hash_
    
    def get_action_index(self, _m, _a):
        for i in range(len(self.a_[_m])):
            # print(self.a_[_m][i].a_, _a)
            if type(_a) is set or type(_a) is list:
                # if self.a_[_m][i].a_ in _a:
                print(self.a_[_m][i].a_, _a)
            if self.a_[_m][i].a_ == _a:
                return i
        return None
    
    def get_policy(self):
        if self.policy_a_ == None:
            return self.m_[0], self.a_[self.m_[0]][0].a_
        return self.policy_m_, self.policy_a_
    
    def set_policy(self, m, a):
        self.policy_m_ = m
        self.policy_a_ = a
    
    def add_child(self, _m, _a, _s_p, _s_p_i, _r):
        """
        Adds child action to state

        Args:
            _a (int): action id 
            _s_p (int): hash key for transition state
            _r (float): reward
        """
        if _m not in self.m_:
            self.m_.append(_m)
            ind = 0
            for i in range(len(self.m_unused_)):
                if self.m_unused_[i] == _m:
                    ind = i
                    break
            self.m_unused_.pop(ind)

            
        # models_with_action = []
        # for m in self.m_:
        #     for a in self.a_[m]:
        #         if _a == a.a_:
        #             models_with_action.append(m)
        # if _m in models_with_action:
        #     child_ind =  self.a_[_m].add_child(_s_p, _s_p_i, _r)
        
        if len(self.a_[_m]) > 0:
            child_ind = -1
            for i in range(len(self.a_[_m])):
                if self.a_[_m][i].a_ == _a:
                    child_ind = self.a_[_m][i].add_child(_s_p, _s_p_i, _r)
                    break
            if child_ind == -1:
                self.a_[_m].append(Action(_a))
                child_ind = self.a_[_m][-1].add_child(_s_p, _s_p_i, _r)
        else:
            self.a_[_m].append(Action(_a))
            child_ind = self.a_[_m][-1].add_child(_s_p, _s_p_i, _r)
        
        self.model_N_[_m] += 1
        self.N_ += 1
        return child_ind
    
    def add_action(self, _m, _a):
        not_found = True
        
        for a in self.a_[_m]:
            if a.a_ == _a:
                not_found = False
                break
        if not_found:        
            self.a_[_m].append(Action(_a))
    # @abstractmethod 
    # def get_N(self):
    #     self.N_ = 0
    #     for a in _self.a_:
    #         self.N_ += a.N_
    
    def print_state(self):
        print("---------PS---------")
        for m in self.m_:
            print("Model: ", m, " N: ", self.model_N_[m])
            for a in self.a_[m]:
                print("-----Action: ", a.a_, " N: ", a.N_, "r: ", a.r_)
    
    def get_transition(self, _m, _a):
        """
        Gets transition model associated with state-action

        Returns:
            list(tuple): transition model associated with state
        """
        for a in self.a_[_m]:
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
        if _s_p_i in self.s_prime_i_:
            for x in range(len(self.s_prime_i_)): 
                if _s_p_i == self.s_prime_i_[x]:
                    i = x
                    break
            self.r_[i] = (self.n_[i]*self.r_[i] + _r)/(self.n_[i]+1)
            self.n_[i] += 1
            ind = self.s_prime_i_[i]
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
        T = np.divide(self.n_, self.N_)
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
        
    @abstractmethod
    def __eq__(self, _a):
        """
        Checks if two actions are equal

        Args:
            _s (State): action to compare against

        Returns:
            bool: true if equal
        """
        return (self.a_ == _a.a_)

    