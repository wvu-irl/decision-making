import copy
from msilib import knownbits
from multiprocessing import parent_process
from pyexpat import model
import random
import numpy as np
import matplotlib.pyplot as plt


from problem.state_action import State, Action
from optimizers.optimization import Optimizer
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import gym


class AOGS():
    """
    Perform Monte Carlo Tree Search 
    Description: User specifies MDP model and AOGS solves the MDP policy to some confidence.
    """
    def __init__(self, _env : gym.Env, _N = 5e3, _bounds = [0, 1], _performance = [0.05, 0.05]): #_action_selection, 

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
        
        #self.as_s_ = _action_selection
        
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
    def reinit(self, _state, _action = None, _s_prime = None):
        """
        Reinitialize Graph from state-action-state transition
        Args:
            self (AOGS): AOGS Object
            _state (State): State Transition
            _action (Action): Action to selected
             
        Returns:
        """
        self.graph_ = [State()] * self.N_
        self.gi : dict = {}
        self.U_ = []
        self.gi_ = []
        self.current_policy = -1
        self.n_ = 0
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
            parents[0] = s
            p_ind = 1
            d = 0
            do_reset = True
            
            while self.graph_[self.gi_[hash(s)]].children and d < _D:   
                if do_reset:     
                    self.env_.reset(s)
                
                #select action argmax upper exp
                    #maybe do epsilon greedy
                
                if self.graph_[s].a_.N_ >= self.t_:
                    s_p, r, done = self.simulate(a)
                    do_reset = False
                else:
                    s_p, r = s.a_.sample_transition_model(self.rng_)
                    do_reset = False
                    
                # add transition to model
                
                if not (hash(s_p) in self.gi_):
                    self.gi_[hash(_s)] = self.n_
                    self.graph_[self.gi_[hash(_s)]] = State(s, self.env_.get_actions(_s))
                    self.n_ += 1
                    
                s = s_p
                parents[p_ind] = s
                p_ind += 1
                
            backpropagate()    
                
    return max over meu   
          
                # 
                # s = _s
                # for t in range(_h):
                #     self.compute_bounds(n)
                #     b = self.select(s,n)
                #     self.graph_eval(obs,reward)
                #     if done:
                #         break
               
    def compute_bounds(self, _n):
        return

    def select(self,_s):
        """
        Select action to take for AOGS Simulation
        Args:
            self (AOGS): AOGS Object
            _s (State): Current State
        Returns:
            Action: Best Action from Current state
        """
        return self.as_s_.return_action(_s,self.a_,self._select_param)

    def simulate(self, _a):
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
        obs, r, done, info = self.env_.step(_a)
        return obs, r, done

    def graph_eval(self, _obs, _r):
        ## IF _Obs is in Graph Already
        ## - Add state, state_ trans, action, reward into  S and new S' node

        ## ELSE
        ##  - Add state, state_ trans, action, reward into  S and S'
        return