"""
The AOGS class implements in the Ambiguous Optimistic Graph Search from 

<insert citation>
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import math
import numpy as np
import time

import gymnasium as gym
import time
from copy import deepcopy

from problem.mm_state_action import State, Action
from select_action.model_selection import model_selection
from select_action.utils import * 

import matplotlib.pyplot as plt


class MM_AAGS(gym.Env):
    """
    Ambiguous Optimistic Graph Search from 

    <insert citation>
    
    
    """
    def __init__(self, _alg_params, _env_params):# _env : gym.Env, _action_selection, _N = 1e5, _bounds = [0, 1], _performance = [0.05, 0.05], _gamma = 0.95): # 

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
        super(MM_AAGS, self).__init__()
        if "search" in _alg_params:
            self.search_params_ = _alg_params["search"]
        _alg_params = _alg_params["params"]; 
        self.alg_params_ = _alg_params
        self.env_params_ = _env_params
        
        self.act_sel_ = action_selection.action_selection(act_sel_funcs[_alg_params["action_selection"]["function"]], _alg_params["action_selection"]["params"])
        self.model_sel_ = model_selection.model_selection(model_sel_funcs[_alg_params["model_selection"]["function"]], _alg_params["model_selection"]["params"])
        
        self.reinit()

        self.m_ = 0
        
        self.t_ = self.num_samples(self.alg_params_["model_accuracy"])

        if "rng_seed" in _alg_params:
            self.rng_ = np.random.default_rng(_alg_params["rng_seed"])
        else:
            self.rng_ = np.random.default_rng()
            
        # self.map_ = np.zeros([self.env_params_["params"]["dimensions"][0],self.env_params_["params"]["dimensions"][0]])

    def num_samples(self, _perf):
        num = math.log( 1/ (2/3*(1-_perf["delta"]+1/2) ) - 1)
        den = math.log( 1/ (2/3*(1-_perf["epsilon"]) ) + 1/2)**2
        t = -num/(_perf["epsilon"]*den)
        # print("t", t)
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
        self.graph_ = [State(1)] * int(self.alg_params_["max_graph_size"])
        self.gi_ : dict = {}
        self.U_ = []
        self.current_policy = -1
        self.n_ = 0
        self.value_gap_ = 1
        self.env_ = gym.make(self.env_params_["mm_env"],max_episode_steps = self.search_params_["horizon"], params=deepcopy(self.env_params_))
        # self.env_.reset()
        # print("===========", self.env_.get_reward_bounds())
        
        self.bounds_ = list(self.env_.get_reward_bounds())
        temp = [self.bounds_[0]*self.search_params_["horizon"], self.bounds_[1]*self.search_params_["horizon"]]
        # print(temp)
        # self.bounds_[0] *= self.search_params_["horizon"]
        # self.bounds_[1] *= self.search_params_["horizon"]
        
    ######################################################
              
    def evaluate(self, _s : State, _search_params = None):
        """
        Conducts Graph search from root
        Args:
            self (AOGS): AOGS Object
            _s (State): State to start search from
            self.search_params_["horizon"] (int) : Max depth to evaluate

        Returns:

        """
        if _search_params != None:
            self.search_params_ = _search_params
        elif "search" in self.alg_params_:
            self.search_params_ = self.alg_params_["search"]
        else:
            self.search_params_ = {
                "max_samples": 1e3,
                "horizon": 5,
                "timeout": 5,
                "reinit": True
            }
        if self.search_params_["reinit"]:
            self.reinit()
                        
        self.m_ = 0
        start_time = time.perf_counter()
        s = None
        self.value_gap_ = self.alg_params_["model_accuracy"]["epsilon"]
        _str_s = hash(str(_s))
        
        # print(self.env_.get_belief())
        if _str_s not in self.gi_:
            self.gi_[_str_s] = self.n_
            self.graph_[self.n_] = State(deepcopy(_s), self.env_.get_belief(), _L= self.bounds_[0], _U = self.bounds_[1])
            self.U_.append(_str_s) 
            self.n_ += 1
            
        self.is_not_converged_ = True
        # print('lol')
        cnt = 0
        while (time.perf_counter()-start_time < self.search_params_["timeout"]) and self.n_ < self.alg_params_["max_graph_size"] and len(self.U_) and self.m_ < self.search_params_["max_samples"] and self.is_not_converged_:
            cnt += 1
            # print("cnt-------",cnt)
            # temp_params = deepcopy(self.env_params_)
            # temp_params["shared"]["state"] = deepcopy(_s)
            temp_params = {"state": deepcopy(_s)}
            # print("------------------------------------")
            # print(_s)
            # self.graph_[self.gi_[_str_s]].print_state()
            # print("------------------------------------")

            s, info = self.env_.reset(options=temp_params)
            
            if _s["objects"] != s["objects"]:
                print("-----------")
                print(_s["objects"])
                print(s["objects"])

            parents = [-1]*int(self.search_params_["horizon"]*5+1)
            p_ind = 0
            self.d_ = 0
            do_reset = True
            is_terminal = False
            # is_leaf = False
            # print("n " + str(self.n_) + ", d " + str(d) )
            #should come up with better way to handle terminal states, check out MCRM
            # right now it may not terminate
            self.is_not_converged_ = False
            while not is_terminal and self.d_ < self.search_params_["horizon"] and self.m_ < self.search_params_["max_samples"]: #check if leaf
                # print("-------d", self.d_)
                str_s = hash(str(s))

                if str_s not in parents:     
                    parents[p_ind] = str_s
                    p_ind += 1

                               
                # print(s)
                a, v_opt, L, U, diffs, exps, _ = self.act_sel_.return_action(self.graph_[self.gi_[str_s]],{"alpha": 1},self)
                # raise NotImplemented("map action space back to global")
                model = a["model"]              
                del a["model"]
                # print(model, a)
                
                # if gap > self.value_gap_:
                #     self.value_gap_ = U-L
                # print("l151 ",s)
                s_p, r, is_terminal, do_reset = self.simulate(s,a,model, do_reset)
                
                if s["objects"] != s_p["objects"]:
                    print("sp-----------")
                    print(_s["objects"])
                    print(s["objects"])
                    print(s_p["objects"])
                # print(r)
                # if a["task"] == 4:
                #     print("repair", r)
                
                str_sp = hash(str(s_p))
                if str_sp in self.gi_:
                    ind = self.gi_[str_sp]
                else:
                    ind = self.n_

                # print(self.graph_[self.gi_[str_s]].s_)
                ind = self.graph_[self.gi_[str_s]].add_child(model, a, s_p, ind,r)
                if ind == self.n_:
                    
                    self.gi_[str_sp] = self.n_
                    if is_terminal:
                        if self.alg_params_["gamma"] == 1:
                            # print(r)
                            v = r*(self.search_params_["horizon"]-self.d_-1)
                            # print(self.d_, v)
                            L = v
                            U = v
                        else:
                            v = r/(1-self.alg_params_["gamma"])         
                            L = v
                            U = v
                    else:
                        v = 0
                    
                    self.graph_[self.gi_[str_sp]] = State(deepcopy(s_p), self.env_.get_belief(), str_s, v, is_terminal, _L = L, _U = U)
                    # print("s ", s_p)
                    # print("act ", self.env_.get_actions(s_p))
                    # for a in self.graph_[self.gi_[str_sp]].a_:
                    #     print(a.a_)
                    self.n_ += 1
                    #is_leaf = True
                    """Took out the is_leaf check. In the spirit of GBOP, they just perform entire trajectories.
                        This seems kind of useful for sparse problems, get's deepe rbut less accurate
                        
                    """
                    if not is_terminal:
                        self.U_.append(str_sp)
                else:
                    if str_s not in self.graph_[self.gi_[str_sp]].parent_:
                        self.graph_[self.gi_[str_sp]].parent_.append(str_s)
                        
                # plt.pause(5)
                    
                self.d_ += 1
                model, policy = self.graph_[self.gi_[str_s]].get_policy()

                pol_ind = self.graph_[self.gi_[str_s]].get_action_index(model, policy)
  
                if str_s in self.U_ and self.graph_[self.gi_[str_s]].a_[model][pol_ind].N_ > self.t_:
                    self.U_.remove(str_s)
                elif str_s not in self.U_ and self.graph_[self.gi_[str_s]].a_[model][pol_ind].N_ <= self.t_: 
                    self.U_.append(str_s)
                
                # if s["objects"] != s_p["objects"]:
                #     print("-----------")
                #     print(s["objects"])
                #     print(s_p["objects"])
                
                s = s_p
            
            # print("s ", hash(str(s)), s)
            # self.graph_[0].print_state()
                
            parents.reverse()   
             
            self.backpropagate(list(set(parents)))
    
        # print("n " + str(self.n_))
        
        ##
        ## For parameters, call get action from env and pass it to the action selection function. 
        ##
        a, e_max, L, U, diffs, exps, pav = self.act_sel_.return_action(self.graph_[self.gi_[_str_s]],{"is_policy" : True},self)
        # raise NotImplemented("map action space back to global")

        print("s", _s["pose"])
        for el in pav:
            print(el)
        # print(pav)
        # print("emax ", e_max)
        # print(exps)
        # print("gap", U-L)
        # print("m ", self.m_)
        # print("n", self.n_)
        # print("g---")
        # plt.cla()
        # for s in self.graph_:
        #     # print(s.s_)
        #     if "pose" in s.s_:
        #         self.map_[s.s_["pose"][0]][s.s_["pose"][1]] +=1
        # print("---g")
        # t_map = (self.map_)
        # print("max map ", np.max(np.max(self.map_)))
        # plt.imshow(np.transpose(t_map), cmap='Reds', interpolation='hanning')
        # plt.pause(1)
        # print(len(self.gi_))
        # print("Usize", len(self.U_))
        
        self.graph_[0].print_state()
        
        
        
        return a#self.graph_[self.gi_[_str_s]].get_action_index(a)
    
    def update_models(self, _s, _a, _s_p, _r):
        self.env_.update_models(_s, _a, _s_p, _r)
               
    def simulate(self, _s, _a, _model, _do_reset):
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
        # print(_a)
        act_ind = self.graph_[self.gi_[hash(str(_s))]].get_action_index(_model, _a)
        # print(_s)
        # print(act_ind)
        if act_ind is None or self.graph_[self.gi_[hash(str(_s))]].a_[_model][act_ind].N_ <= self.t_:
            # self.map_[_s["pose"][0]][_s["pose"][1]] += 1
            if _do_reset: 
                # temp_params = deepcopy(self.env_params_)
                # temp_params["shared"]["state"] = deepcopy(_s)
                temp_params = {"state": deepcopy(_s)}
                # gym.make(self.env_params_["env"],max_episode_steps = self.search_params_["horizon"], _params=self.env_params_["params"])
                # print(_s)
                s, info = self.env_.reset(options=temp_params)
                # print("sa", s)
                # print(_s, self.env_.get_observation())
            # print(_a)
            # print(self.env_.step(_a))
            # exit()
            # model = self.model_sel_.model_selection(self.graph_[self.gi_[str_s]],self.alg_params_["model_selection"]["params"],self)
            start_model_time = time.perf_counter()
            s_p, r, done, is_trunc, info = self.env_.step({"model": _model, **_a})
            stop_model_time = time.perf_counter()
                
            self.env_.align_resource_usage(_model,stop_model_time-start_model_time)
            done = done or is_trunc
            self.m_+=1
            # self.d_+=1
            _do_reset = False
            self.is_not_converged_ = True
            # if s_p == {"pose":[10,10]}:
            #     print(done)
            # if r >= 0:
            #     print(_s,s_p,r)
        else:
            s_p, r = self.graph_[self.gi_[hash(str(_s))]].a_[_model][act_ind].sample_transition_model(self.rng_)
            done = self.graph_[self.gi_[hash(str(s_p))]].is_terminal_
            _do_reset = True
            # self.d_+=0.2
            # self.is_converged_ = self.is_converged_ and True
        # print(s_p)
        return s_p, r, done, _do_reset
    
    def backpropagate(self, _parents):
        
        start_time = time.perf_counter()
        while len(_parents) and (time.perf_counter()-start_time < 10):
            #print(_parents)
            s = _parents.pop(0)
            #print("lp " + str(len(_parents)))
            if s != -1:
                                                                                                    #, "alpha": 1                        
                a, v, L, U, diffs, exps, _ = self.act_sel_.return_action(self.graph_[self.gi_[s]],{"is_policy" : True},self)
                # raise NotImplemented("map action space back to global")

                if self.alg_params_["gamma"] == 1:
                    lprecision = diffs[0]*self.alg_params_["diffs_const"]
                    uprecision = diffs[1]*self.alg_params_["diffs_const"]
                else:
                    lprecision = (1-self.alg_params_["gamma"])/self.alg_params_["gamma"]*diffs[0]
                    # print("----------")
                    # print(lprecision)
                    # print(np.abs(L - self.graph_[self.gi_[s]].L_))
                    # print(np.abs(L - self.graph_[self.gi_[s]].L_) > lprecision)
                    uprecision = (1-self.alg_params_["gamma"])/self.alg_params_["gamma"]*diffs[1]
                    # print(uprecision)
                    # print(np.abs(U - self.graph_[self.gi_[s]].U_))
                    # print(np.abs(U - self.graph_[self.gi_[s]].U_) > uprecision)
                if np.abs(U - self.graph_[self.gi_[s]].U_) > uprecision or np.abs(L - self.graph_[self.gi_[s]].L_) > lprecision:
                    temp = self.graph_[self.gi_[s]].parent_
                    for p in temp:
                        if p not in _parents:
                            _parents.append(p)
                self.graph_[self.gi_[s]].V_ = v
                self.graph_[self.gi_[s]].L_ = L
                self.graph_[self.gi_[s]].U_ = U
                temp_a = deepcopy(a)
                del temp_a["model"]
                self.graph_[self.gi_[s]].set_policy(a["model"], temp_a)
                # print("V", v)
                # print("L", L)
                # print("U", U)
                # lmn = 0
        _parents.clear()

            
        