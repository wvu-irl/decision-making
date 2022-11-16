import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import time
import copy

import sys
import os


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from problem.state_action import State, Action
from select_action.utils import * 


class GBOP():
    """
    Perform Monte Carlo Tree Search 
    Description: User specifies MDP model and GBOP solves the MDP policy to some confidence.
    """
    def __init__(self, _alg_params, _env_params): #
        """
         Constructor, initializes BMF-AST
         Args:
             self (GBOP): GBOP Object
             _mdp (MDP): Markov Decision Process to solve
             _N (int): Number of nodes to process in tree
             _n_rollout (int): Maximum number of rollout iterations
         Returns:
             GBOP: GBOP object
         """
        super(GBOP, self).__init__()

        self.alg_params_ = _alg_params
        self.env_params_ = _env_params
        if "search" in _alg_params:
            self.search_params_ = _alg_params["search"]
        self.alpha_ = _alg_params["action_selection"]["move_params"]["alpha"]
            
        self.a_s_b_ = action_selection.action_selection(act_sel_funcs[_alg_params["action_selection"]["bound_function"]], _alg_params["action_selection"]["bound_params"])
        self.a_s_m_ = action_selection.action_selection(act_sel_funcs[_alg_params["action_selection"]["move_function"]], _alg_params["action_selection"]["move_params"])
        
        self.bounds_ = [_env_params["params"]["reward_bounds"][0]/(1-_alg_params["gamma"]), _env_params["params"]["reward_bounds"][1]/(1-_alg_params["gamma"])]

        self.m_ = 0
        
        self.reinit()
        
        if "rng_seed" in _alg_params:
            self.rng_ = np.random.default_rng(_alg_params["rng_seed"])
        else:
            self.rng_ = np.random.default_rng()
            
        self.map_ = np.zeros([self.env_params_["params"]["dimensions"][0],self.env_params_["params"]["dimensions"][0]])

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
        
        self.graph_ = [State(1)] * int(self.alg_params_["max_graph_size"])
        self.gi_ : dict = {}
        self.current_policy = -1
        self.n_ = 0
        self.value_gap_ = 1
        self.env_ = gym.make(self.env_params_["env"],max_episode_steps = self.search_params_["horizon"], _params=self.env_params_["params"])

    ######################################################

    def search(self, _s : State, _search_params = None):              
        """
        Conducts Graph search from root
        Args:
            self (GBOP): GBOP Object
            _s (State): State to start search from
            _h (int) : Horizon to evaluate

        Returns:

        """
        if _search_params != None:
            self.search_params_ = _search_params
        elif "search" not in self.alg_params_:
            self.search_params_ = {
                "branch_factor_s": 10,
                "branch_factor_a": 10,
                "max_samples": 1e3,
                "horizon": 5,
                "timeout": 5,
                "reinit": True
            }
        if self.search_params_["reinit"]:
            self.reinit()
                
        self.m_ = 0
        start_time = time.perf_counter()
        _str_s = hash(str(_s))
        
        if _str_s not in self.gi_:
            self.gi_[_str_s] = self.n_
            self.graph_[self.n_] = State(_s, self.env_.get_actions(_s), _L= self.bounds_[0], _U = self.bounds_[1])
            self.n_ += 1
        
        #N is the number of trajectories now    
        while (time.perf_counter()-start_time < self.search_params_["timeout"]) and self.n_ < self.alg_params_["max_graph_size"] and self.m_ < self.search_params_["max_samples"]:
            
            self.env_ = gym.make(self.env_params_["env"],max_episode_steps = (self.search_params_["horizon"]*2), _params=self.env_params_["params"])
            self.env_.reset()
            
            s = _s
            
            self.d_ = 0
            is_terminal = False
            
            while not is_terminal and self.d_ < self.search_params_["horizon"] and self.m_ < self.search_params_["max_samples"]:
                
                str_s = hash(str(s))
                L, U = self.bound_outcomes(str_s)

                a = self.a_s_m_.return_action(self.graph_[self.gi_[str_s]],[1],self)

                s_p, r, is_terminal = self.simulate(s, a)

                str_sp = hash(str(s_p))
                if str_sp in self.gi_:
                    ind = self.gi_[str_sp]
                else:
                    ind = self.n_
                ind = self.graph_[self.gi_[str_s]].add_child(a, s_p, ind,r)

                if ind == self.n_:
                    
                    self.gi_[str_sp] = self.n_
                    if is_terminal:
                        v = r/(1-self.alg_params_["gamma"])
                        L = v
                        U = v
                    else:
                        v = 0
                    self.graph_[self.gi_[str_sp]] = State(s_p, self.env_.get_actions(s_p), str_s, v, is_terminal, _L = L, _U = U)
                    self.n_ += 1
                else:
                    self.graph_[self.gi_[str_sp]].parent_.append(str_s)
                self.d_ += 1
                s = s_p
                # print(t)
                # print(self.n_)
        plt.cla()
        for s in self.graph_:
            # print(s.s_)
            if "pose" in s.s_:
                self.map_[s.s_["pose"][0]][s.s_["pose"][1]] +=1
        t_map = (self.map_)
        print("max map ", np.max(np.max(self.map_)))
        plt.imshow(np.transpose(t_map), cmap='Reds', interpolation='hanning')
        plt.pause(1)
        a = self.a_s_m_.return_action(self.graph_[self.gi_[_str_s]],[self.alpha_],self)
        return a
               
    def bound_outcomes(self, _s):
        parents = [_s]
        while len(parents):
            s = parents.pop(0)
            t =0
            if s != -1:
                L,U = self.a_s_b_.return_action(self.graph_[self.gi_[s]],[self.alpha_],self)
                
                lprecision = ((1-self.alg_params_["gamma"])/self.alg_params_["gamma"])*self.alg_params_["model_accuracy"]["epsilon"]
                uprecision = ((1-self.alg_params_["gamma"])/self.alg_params_["gamma"])*self.alg_params_["model_accuracy"]["epsilon"]

                if np.abs(U - self.graph_[self.gi_[s]].U_) > uprecision or np.abs(L - self.graph_[self.gi_[s]].L_) > lprecision:
                    temp = self.graph_[self.gi_[s]].parent_
                    for p in temp:
                        if p not in parents:
                            parents.append(p)
                self.graph_[self.gi_[s]].L_ = L
                self.graph_[self.gi_[s]].U_ = U
                t += 1
        return L, U 
                

    def select(self,_s):
        """
        Select action to take for GBOP Simulation
        Args:
            self (GBOP): GBOP Object
            _s (State): Current State
        Returns:
            Action: Best Action from Current state
        """
        return self.as_s_.return_action(_s,self.a_,self._select_param)

    def simulate(self, _s, _a):
        """
        Simulate the GBOP object's Enviroment
        Args:
            self (GBOP): GBOP Object
            _a (State): Action to take
        Returns:
            obs: Observation of simulation
            r: reward collected from simulation
            done (bool): Flag for simulation completion
        """
        self.map_[_s["pose"][0]][_s["pose"][1]] += 1
        s_p, r, done, info = self.env_.step(_a)
        self.m_+=1

        return s_p, r, done


