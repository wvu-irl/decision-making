import random
import math
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

class AOGS():
    """
    Perform Monte Carlo Tree Search 
    Description: User specifies MDP model and AOGS solves the MDP policy to some confidence.
    """
    def __init__(self, _env : gym.Env, _action_selection, _N = 1e5, _bounds = [0, 1], _performance = [0.05, 0.05], _gamma = 0.95): # 

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
        self.bounds_ = [_bounds[0]/(1-_gamma), _bounds[1]/(1-_gamma)]
        self.gamma_ = _gamma
        
        self.a_s_ = _action_selection

        self.reinit()
        
        self.t_ = self.num_samples(_performance)

        self.rng_ = np.random.default_rng()

    def num_samples(self, _perf):
        num = math.log( 1/ (2/3*(1-_perf[1]+1/2) ) - 1)
        den = math.log( 1/ (2/3*(1-_perf[0]) ) + 1/2)**2
        t = -num/(_perf[0]*den)
        print("t", t)
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
        self.value_gap_ = 1
    ######################################################
              
    def search(self, _s : State, _D :int = 100, _timeout = 10, _reinit = False):
        """
        Conducts Graph search from root
        Args:
            self (AOGS): AOGS Object
            _s (State): State to start search from
            _D (int) : Max depth to evaluate

        Returns:

        """
        if _reinit:
            self.reinit()
        start_time = time.perf_counter()
        s = None
        self.value_gap_ = self.performance_[0]
        _str_s = hash(str(_s))
        self.m_ = 0
        if _str_s not in self.gi_:
            self.gi_[_str_s] = self.n_
            # print("act ", self.env_.get_actions(_s))
            self.graph_[self.n_] = State(_s, self.env_.get_actions(_s))
            
            self.U_.append(_str_s) 
            self.n_ += 1
        
        while (time.perf_counter()-start_time < _timeout) and self.n_ < self.N_ and len(self.U_) and self.m_ < 5460:
            # print("------------")
            # for i in range(len(self.gi_)):
            #     print(self.graph_[i].s_)
            # print(self.gi_)    
            if _str_s not in self.U_:
                # print("nee")
                s = self.graph_[self.gi_[self.rng_.choice(self.U_)]].s_
            else:
                # print("yee")
                s = _s
            
            parents = [-1]*_D
            p_ind = 0
            d = 0
            do_reset = True
            is_terminal = False
            is_leaf = False
            # print("n " + str(self.n_) + ", d " + str(d) )
            #should come up with better way to handle terminal states, check out MCRM
            # right now it may not terminate
            
            while not is_leaf and not is_terminal and d < _D:
                
                # print("n " + str(self.n_) + ", d " + str(d) )
                # print("s ", s)
                str_s = hash(str(s))
                if str_s not in parents:     
                    parents[p_ind] = str_s
                    p_ind += 1
                # print(str_s)
                #pass alpha into initialization, 
                # bounds and params available from solver 
                a, v_opt, L, U, exps = self.a_s_.return_action(self.graph_[self.gi_[str_s]],[1],self)
                
                # if gap > self.value_gap_:
                #     self.value_gap_ = U-L
                # print("l151 ",s)
                s_p, r, is_terminal, do_reset = self.simulate(s,a, do_reset)
                # print(r)
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
                        L = v
                        U = v
                    else:
                        v = 0
                    self.graph_[self.gi_[str_sp]] = State(s_p, self.env_.get_actions(s_p), str_s, v, is_terminal, _L = L, _U = U)
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
                    
                d += 1
                
                remove_u = True
                for a in self.graph_[self.gi_[str_s]].a_:
                    remove_u = remove_u and (a.N_ > self.t_)
                if str_s in self.U_ and remove_u:
                    self.U_.remove(str_s)
                
                s = s_p
            
            parents.reverse()   
             
            self.backpropagate(list(set(parents)))
    
        print("n " + str(self.n_))
        a, e_max, L, U, exps = self.a_s_.return_action(self.graph_[self.gi_[_str_s]],[],self)
        print("emax ", e_max)
        print(exps)
        print("gap", U-L)
        #print("m ", self.m_)
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
        # print(_a)
        act_ind = self.graph_[self.gi_[hash(str(_s))]].get_action_index(_a)
        # print(_s)
        # print(act_ind)
        if self.graph_[self.gi_[hash(str(_s))]].a_[act_ind].N_ <= self.t_:
            if _do_reset:     
                self.env_.reset(_s)
            s_p, r, done, info = self.env_.step(_a)
            #self.m_+=1
            _do_reset = False
        else:
            s_p, r = self.graph_[self.gi_[hash(str(_s))]].a_[act_ind].sample_transition_model(self.rng_)
            done = self.graph_[self.gi_[hash(str(s_p))]].is_terminal_
            _do_reset = True
        return s_p, r, done, _do_reset 
    
    def backpropagate(self, _parents):
        
        
        while len(_parents):
            #print(_parents)
            s = _parents.pop(0)
            #print("lp " + str(len(_parents)))
            if s != -1:
                a, v, L, U, exps = self.a_s_.return_action(self.graph_[self.gi_[s]],[],self)
                lprecision = (1-self.gamma_)/self.gamma_*exps[0]*60
                # print("----------")
                # print(lprecision)
                # print(np.abs(L - self.graph_[self.gi_[s]].L_))
                # print(np.abs(L - self.graph_[self.gi_[s]].L_) > lprecision)
                uprecision = (1-self.gamma_)/self.gamma_*exps[1]*60
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
                # print("V", v)
                # print("L", L)
                # print("U", U)

            
        