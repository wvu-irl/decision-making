import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import time
from copy import deepcopy
import sys
import os


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from problem.state_action import State, Action
from select_action.utils import * 

import gymnasium as gym

class GBOP(gym.Env):
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

        if "search" in _alg_params:
            self.search_params_ = _alg_params["search"]
        _alg_params = _alg_params["params"]; 
        self.alg_params_ = _alg_params
        self.env_params_ = _env_params
        
        self.alpha_ = _alg_params["action_selection"]["move_params"]["alpha"]
        self.a_s_b_ = action_selection.action_selection(act_sel_funcs[_alg_params["action_selection"]["bound_function"]], _alg_params["action_selection"]["bound_params"])
        self.a_s_m_ = action_selection.action_selection(act_sel_funcs[_alg_params["action_selection"]["move_function"]], _alg_params["action_selection"]["move_params"])
        
        self.bounds_ = [_env_params["params"]["r_range"][0]/(1-_alg_params["gamma"]), _env_params["params"]["r_range"][1]/(1-_alg_params["gamma"])]

        self.m_ = 0
        
        self.reinit()
        
        if "rng_seed" in _alg_params:
            self.rng_ = np.random.default_rng(_alg_params["rng_seed"])
        else:
            self.rng_ = np.random.default_rng()
            
        # self.map_ = np.zeros([self.env_params_["params"]["dimensions"][0],self.env_params_["params"]["dimensions"][0]])

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
        self.env_ = gym.make(self.env_params_["env"],max_episode_steps = self.search_params_["horizon"], params=deepcopy(self.env_params_["params"]))

    ######################################################

    def evaluate(self, _s : State, _search_params = None):              
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
        elif "search" in self.alg_params_:
            self.search_params_ = self.alg_params_["search"]
        else:
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
            a_list, neighbors = self.env_.get_actions(_s)
            self.graph_[self.n_] = State(deepcopy(_s), a_list, _L= self.bounds_[0], _U = self.bounds_[1])
            self.n_ += 1
        # print("-----------")
        #N is the number of trajectories now    
        while (time.perf_counter()-start_time < self.search_params_["timeout"]) and self.n_ < self.alg_params_["max_graph_size"] and self.m_ < self.search_params_["max_samples"]:
            
            temp_params = deepcopy(self.env_params_)
            temp_params["params"]["state"] = deepcopy(_s)
            # print(_s)
            # gym.make(self.env_params_["env"],max_episode_steps = self.search_params_["horizon"], _params=self.env_params_["params"])
            # self.env_ = gym.make(self.env_params_["env"],max_episode_steps = (self.search_params_["horizon"]*2), _params=self.env_params_["params"])
            s, info = self.env_.reset(options=temp_params["params"])
                        
            self.d_ = 0
            is_terminal = False
            
            while not is_terminal and self.d_ < self.search_params_["horizon"] and self.m_ < self.search_params_["max_samples"]:
                # print(s)
                
                
                str_s = hash(str(s))
                # print(s, str_s)
                L, U = self.bound_outcomes(str_s)
                
                # if (s["pose"] == [20,20]):
                #     print(s)
                #     print(L,U)
                #     plt.pause(0.05)

                a = self.a_s_m_.return_action(self.graph_[self.gi_[str_s]],[1],self)

                s_p, r, is_terminal = self.simulate(s, a)
                # print("       ",s_p, hash(str(s_p)))
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
                    # temp_sp = {}
                    # for el in s_p:
                    #     temp_sp[el] = deepcopy(s_p[el])
                    #     print(s_p[el])
                    a_list, neighbors = self.env_.get_actions(s_p)
                    self.graph_[self.gi_[str_sp]] = State(deepcopy(s_p), a_list, str_s, v, is_terminal, _L = L, _U = U)
                    self.n_ += 1
                    # print("save_par", self.graph_[self.gi_[str_s]].s_, hash(str(self.graph_[self.gi_[str_sp]].s_)))
                    # print("save", self.graph_[self.gi_[str_sp]].s_, hash(str(self.graph_[self.gi_[str_sp]].s_)))
                else:
                    self.graph_[self.gi_[str_sp]].parent_.append(str_s)
                self.d_ += 1
                s = s_p
                # print(t)
                # print(self.n_)
        # plt.cla()
        # for s in self.graph_:
        #     # print(s.s_)
        #     if "pose" in s.s_:
        #         self.map_[s.s_["pose"][0]][s.s_["pose"][1]] +=1
        # t_map = (self.map_)
        # print("max map ", np.max(np.max(self.map_)))
        # plt.imshow(np.transpose(t_map), cmap='Reds', interpolation='hanning')
        # plt.pause(1)
        a = self.a_s_m_.return_action(self.graph_[self.gi_[_str_s]],[self.alpha_],self)
        return a
               
    def bound_outcomes(self, _s):
        parents = [_s]
        start_time = time.perf_counter()
        while len(parents) and (time.perf_counter()-start_time < 10):
            s = parents.pop(0)
            t =0
            if s != -1:
                # print(s)
                # print(self.gi_[s])

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
            count += 1
        return L, U 

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
        # self.map_[_s["pose"][0]][_s["pose"][1]] += 1
        s_p, r, done, is_trunc, info = self.env_.step(_a)
        done = done or is_trunc
        self.m_+=1
        return s_p, r, done


