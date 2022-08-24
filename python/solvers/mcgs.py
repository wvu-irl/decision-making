import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import time

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from problem.state_action import State, Action
from optimizers.optimization import Optimizer


class MCGS():
    """
    Perform Monte Carlo Tree Search 
    Description: User specifies MDP model and MCTS solves the MDP policy to some confidence.
    """
    def __init__(self, _env : gym.Env, _action_selection, _N = 1e5, _bounds = [0, 1], _performance = [0.05, 0.05], _gamma = 0.95): # 

        """
         Constructor, initializes BMF-AST
         Args:
             self (MCTS): MCTS Object
             _mdp (MDP): Markov Decision Process to solve
             _N (int): Number of nodes to process in tree
             _n_rollout (int): Maximum number of rollout iterations
         Returns:
             MCTS: MCTS object
         """
        super(MCGS, self).__init__()

        self.env_ = _env
        self.bounds_ = _bounds
        self.N_ = int(_N)
        self.performance_ = _performance
        self.bounds_ = [_bounds[0]/(1-_gamma), _bounds[1]/(1-_gamma)]
        self.gamma_ = _gamma
        
        self.a_s_ = _action_selection
    
        self.reinit()

        self.m_ = 0


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
        self.gi_ : dict = {}
        self.current_policy = -1
        self.n_ = 0
        self.value_gap_ = self.performance_[0]
    ######################################################
              
    def search(self, _s : State, _D :int = 100, _timeout = 10, _reinit = False):
        """
        Conducts Graph search from root
        Args:
            self (MCGS): MCGS Object
            _s (State): State to start search from
            _h (int) : Horizon to evaluate

        Returns:

        """
        if _reinit:
            self.reinit()
        start_time = time.perf_counter()

        _str_s = hash(str(_s))
        if _str_s not in self.gi_:
            self.gi_[_str_s] = self.n_
            # print("act ", self.env_.get_actions(_s))
            self.graph_[self.n_] = State(_s, self.env_.get_actions(_s))
            self.n_ += 1
        
        #N is the number of trajectories now    
        while (time.perf_counter()-start_time < _timeout) and self.n_ < self.N_:
            s = _s
            str_s = hash(str(s))
            self.env_.reset()
            
            d = 0
            do_reset = True
            is_terminal = False
            is_leaf = False
            
            while not is_leaf and not is_terminal and d < _D:
                
                """compute bounds
                """
                #do optimistic action selection
                a, v_opt, gap, exps = self.a_s_.return_action(self.graph_[self.gi_[str_s]],[1],self)
                
                s_p, r, is_terminal, do_reset = self.simulate(s,a, do_reset)

                str_sp = hash(str(s_p))
                if str_sp in self.gi_:
                    ind = self.gi_[str_sp]
                else:
                    ind = self.n_

                ind = self.graph_[self.gi_[str_s]].add_child(a, s_p, ind,r)
                if ind == self.n_:
                    
                    self.gi_[str_sp] = self.n_
                    if is_terminal:
                        v = r/(1-self.gamma_)
                    else:
                        v = 0
                    self.graph_[self.gi_[str_sp]] = State(s_p, self.env_.get_actions(s_p), str_s, v, is_terminal)
                    # print("s ", s_p)
                    # print("act ", self.env_.get_actions(s_p))
                    # for a in self.graph_[self.gi_[str_sp]].a_:
                    #     print(a.a_)
                    self.n_ += 1
                    #is_leaf = True
                    """Took out the is_leaf check. In the spirit of GBOP, they just perform entire trajectories.
                        This seems kind of useful for sparse problems, get's deepe rbut less accurate
                        
                    """
                else:
                    self.graph_[self.gi_[str_sp]].parent_.append(str_s)
                    
                d += 1
                s = s_p
                
        print("n " + str(self.n_))
        a, e_max, gap, exps = self.a_s_.return_action(self.graph_[self.gi_[_str_s]],[0],self)
        print("emax ", e_max)
        print(exps)
        print("gap", gap)
        print("m ", self.m_)
        return a
               
    def compute_bounds(self, _n):
        return

    def select(self,_s):
        """
        Select action to take for MCGS Simulation
        Args:
            self (MCGS): MCGS Object
            _s (State): Current State
        Returns:
            Action: Best Action from Current state
        """
        return self.as_s_.return_action(_s,self.a_,self._select_param)

    def simulate(self, _s, _a):
        """
        Simulate the MCGS object's Enviroment
        Args:
            self (MCGS): MCGS Object
            _a (State): Action to take
        Returns:
            obs: Observation of simulation
            r: reward collected from simulation
            done (bool): Flag for simulation completion
        """
        act_ind = self.graph_[self.gi_[hash(str(_s))]].get_action_index(_a)

        s_p, r, done, info = self.env_.step(act_ind)
        self.m_+=1

        return s_p, r, done

