import copy
#from msilib import knownbits
#from multiprocessing import parent_process
from pyexpat import model
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import gym

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from problem.state_action import State, Action
from optimizers.optimization import Optimizer




class AOGS():
    """
    Perform Monte Carlo Tree Search 
    Description: User specifies MDP model and AOGS solves the MDP policy to some confidence.
    """
    def __init__(self, _env : gym.Env, _action_selection, _N = 5e3, _bounds = [0, 1], _performance = [0.05, 0.05], _gamma = 0.95): # 

        """
         Constructor, initializes BMF-AST
         Args:
             self (AOGS): AOGS Object
             _env (gym.Env): OpenAI Gym environment
             _N (int): Number of nodes to process in tree
             _n_rollout (int): Maximum number of rollout iterations
         Returns:
             AOGS: AOGS object
         """
        super(AOGS, self).__init__()

        self.env_ = _env
        self.bounds_ = _bounds
        self.N_ = int(_N)
        self.performance_ = _performance
        self.bounds_ = _bounds
        self.gamma_ = _gamma
        
        self.a_s_ = _action_selection
        
        #### COME BACK ######
        a = _env.action_space #Action Space
        self.a_ = a 
        self._select_param = []
        #####################

        self.reinit()
        # for i in range(_N):
        #     self.graph_[i] = State()
        
        self.t_ = self.num_samples(_performance)

        self.rng_ = np.random.default_rng()

    def num_samples(self, _perf):
        num = math.log( 1/ (2/3*(1-_perf[1]+1/2) ) - 1)
        den = math.log( 1/ (2/3*(1-_perf[0]) ) + 1/2)**2
        t = -num/(_perf[0]*den)
        return t
    ############## COME BACK ##############################
    def reinit(self, _state = None, _action = None, _s_prime = None):
        """
        Reinitialize Graph from state-action-state transition
        Args:
            self (AOGS): AOGS Object
            _state (State): State Transition
            _action (Action): Action to selected
             
        Returns:
        """
        self.graph_ = [State(1)] * self.N_
        self.gi_ : dict = {}
        self.U_ = []
        self.current_policy = -1
        self.n_ = 0
        self.value_gap_ = self.performance_[0]
    ######################################################
              
    def search(self, _s : State, _D :int = 100):
        """
        Conducts Graph search from root
        Args:
            self (AOGS): AOGS Object
            _s (State): State to start search from
            _D (int) : Max depth to evaluate

        Returns:

        """
        n = 0
        s = None
        self.value_gap_ = self.performance_[0]
        _str_s = hash(str(_s))
        
        if self.n_ == 0:
            self.gi_[_str_s] = self.n_
            self.graph_[self.gi_[_str_s]] = State(s, self.env_.get_actions(_s))
            
            self.U_.append(_str_s) 
            self.n_ = 1
        
        while (n < self.N_ and len(self.U_)):
            if not (_str_s in self.U_):
                s = self.rng_.choice(self.U_)
            else:
                s = _s
            
            parents = [-1]*_D
            p_ind = 1
            d = 0
            do_reset = True
            is_terminal = False
            is_leaf = False
            str_s = hash(str(s))
            
            #should come up with better way to handle terminal states, check out MCRM
            # right now it may not terminate

            while not is_leaf and not is_terminal and d < _D:
                if str_s not in parents:     
                    parents[p_ind] = str_s
                
                #pass alpha into initialization, 
                # bounds and params available from solver 
                print(type(self))
                print(self)
                a, v_opt, gap = self.a_s_.return_action(s,[1, None],self)
                
                if gap > self.value_gap_:
                    self.value_gap_ = gap
                
                s_p, r, is_terminal, do_reset = self.simulate(s,a, do_reset)
                    
                ind = self.graph_[self.gi_[_str_s]].add_child(s_p, self.n_,r)
                
                if ind == self.n_:
                    str_sp = hash(str(s_p))
                    self.gi_[str_sp] = self.n_
                    self.graph_[self.gi_[str_sp]] = State(s_p, self.env_.get_actions(s_p), str_s, r/(1-self.gamma_), is_terminal)
                    self.n_ += 1
                    is_leaf = True
                    if not is_terminal:
                        self.U_.append(str_sp)
                    
                s = s_p
                p_ind += 1
                
            self.backpropagate(parents)  
            
        a, gap = self.a_s_.select_action(_s,None,None,self)
        return a
               
    def simulate(self, _s, _a, _do_reset):
        """
        Simulate the AOGS object's Enviroment
        Args:
            self (AOGS): AOGS Object
            _a (State): Action to take
        Returns:
            obs: Observation of simulation
            r: reward collected from simulation
            done (bool): Flag for simulation completion
        """
        if _do_reset:     
            self.env_.reset(_s)
            
        if self.graph_[self.gi_[hash(str(_s))]].a_.N_ >= self.t_:
            s_p, r, done, info = self.env_.step(_a)
            _do_reset = False
        else:
            s_p, r = _s.a_.sample_transition_model(self.rng_)
            _do_reset = True
        return s_p, r, done, _do_reset 
    
    def backpropagate(self, _parents):
        precision = (1-self.gamma_)/self.gamma_*self.value_gap_
        
        while len(_parents):
            s = _parents.pop(0)
            if s != -1:
                a, v, gap = self.a_s_.return_action(s,[],self)
                if v - self.graph_[s].V_ > precision:
                    temp = self.graph_[s].parent_
                    for p in temp:
                        if p not in _parents:
                            _parents.append(p)
                self.graph_[s].V_ = v

            
        