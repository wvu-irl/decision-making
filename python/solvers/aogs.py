import copy
#from msilib import knownbits
#from multiprocessing import parent_process
from pyexpat import model
import random
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
        self.gi : dict = {}
        self.U_ = []
        self.gi_ = []
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
        
        if self.n_ == 0:
            self.gi_[hash(_s)] = self.n_
            self.graph_[self.gi_[hash(_s)]] = State(s, self.env_.get_actions(_s))
            
            self.U_.append(hash(_s)) 
            self.n_ = 1
        
        while (n < self.N_ and len(self.U_)):
            if not (_s in self.U_):
                s = self.rng_.choice(self.U_)
            else:
                s = _s
            
            parents = [-1]*_D
            p_ind = 1
            d = 0
            do_reset = True
            is_terminal = False
            
            #should come up with better way to handle terminal states, check out MCRM
            # right now it may not terminate
            while self.graph_[self.gi_[hash(s)]].children and is_terminal and d < _D:
                if hash(s) not in parents:     
                    parents[p_ind] = hash(s)
                
                #pass alpha into initialization, 
                # bounds and params available from solver 
                a, v_opt, gap = self.a_s_.select_action(s,None,[1],self)
                
                if gap > self.value_gap_:
                    self.value_gap_ = gap
                
                s_p, r, is_terminal, do_reset = self.simulate(s,a, do_reset)
                    
                ind = self.graph_[self.gi_[hash(_s)]].add_child(s_p, self.n_,r)
                
                if ind == self.n_:
                    self.gi_[hash(s_p)] = self.n_
                    self.graph_[self.gi_[hash(s_p)]] = State(s_p, self.env_.get_actions(s_p), hash(s), r/(1-self.gamma_), is_terminal)
                    self.n_ += 1
                    if not is_terminal:
                        self.U_.append(hash(s_p))
                    
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
            
        if self.graph_[self.gi_[hash(_s)]].a_.N_ >= self.t_:
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
                a, v, gap = self.a_s_.select_action(s,None,[],self)
                if v - self.graph_[s].V_ > precision:
                    temp = self.graph_[s].parent_
                    for p in temp:
                        if p not in _parents:
                            _parents.append(p)
                self.graph_[s].V_ = v

            
        