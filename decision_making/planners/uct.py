import copy
from email.mime import base
import random
from unittest.case import _BaseTestCaseContext
import numpy as np
import matplotlib.pyplot as plt
import time


import sys
import os
import copy

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# from optimizers.optimization import Optimizer, Bellman

from problem.state_action import State, Action
from select_action.utils import * 

import gym

class UCT():
    """
    Perform Upper Confidence Tree Search 
    Description: User specifies problem and UCT solves for the policy
    """
    def __init__(self, _alg_params, _env_params):# _env : gym.Env, _action_selection, _N = 1e5, _bounds = [0, 1], _performance = [0.05, 0.05], _gamma = 0.95):\
        """
         Constructor, initializes BMF-AST
         Args:
             self (UCT): UCT Object
             _opt (Optimizer): Markov Decision Process to solve
             _env (Gym): Gym environment
             _N (int): Number of nodes to process in tree
             _c (double): exploration/exploitation term (0: total Exploitation, 1: Mostly Exploration)
             _n_rollout (int): Maximum number of rollout iterations_select_action_selection_
         Returns:
             UCT: UCT object
         """
        super(UCT, self).__init__()

        self.alg_params_ = _alg_params
        self.env_params_ = _env_params
        if "search" in _alg_params:
            self.search_params_ = _alg_params["search"]

        self.as_s_ = action_selection.action_selection(act_sel_funcs[_alg_params["action_selection"]["decision_function"]], _alg_params["action_selection"]["decision_params"])
        self.as_r_ = action_selection.action_selection(act_sel_funcs[_alg_params["action_selection"]["rollout_function"]], _alg_params["action_selection"]["rollout_params"])

        self.bounds_ = [_env_params["params"]["reward_bounds"][0]/(1-_alg_params["gamma"]), _env_params["params"]["reward_bounds"][1]/(1-_alg_params["gamma"])]

        self.m_ = 0

        self.reinit()

        if "rng_seed" in _alg_params:
            self.rng_ = np.random.default_rng(_alg_params["rng_seed"])
        else:
            self.rng_ = np.random.default_rng()
            
        self.map_ = np.zeros([self.env_params_["params"]["dimensions"][0],self.env_params_["params"]["dimensions"][0]])        
    
    def reinit(self):
        self.tree_ = [State(1)] * int(self.alg_params_["max_graph_size"])
        self.n_ = 0   
        self.render_ = False
        self.terminalStates = []
        self.terminalActions = [] 
        self.env_ = gym.make(self.env_params_["env"],max_episode_steps = self.search_params_["horizon"], _params=self.env_params_["params"])


    # def learn(self, s_ , _num_samples = 5e3, budget : int = 5000,gamma = .9):
    def search(self, _s : State, _search_params = None):
        
        if _search_params != None:
            self.search_params_ = _search_params
        elif "search" in self.alg_params_:
            self.search_params_ = self.alg_params_["search"]
        else:
            self.search_params_ = {
                "rollout": 100,
                "max_samples": 1e3,
                "horizon": 5,
                "timeout": 5,
                "reinit": True
            }
        if self.search_params_["reinit"]:
            self.reinit()

        self.m_ = 0
        self.n_ += 1
        start_time = time.perf_counter()
        s = _s
        
        self.tree_[0] = State(_s,self.env_.get_actions(_s),None,0)

        while (time.perf_counter()-start_time < self.search_params_["timeout"]) and self.n_ < self.alg_params_["max_graph_size"] and self.m_ < self.search_params_["max_samples"]:
            temp_params = copy.deepcopy(self.env_params_)
            temp_params["params"]["state"] = _s["pose"]
            # gym.make(self.env_params_["env"],max_episode_steps = self.search_params_["horizon"], _params=self.env_params_["params"])
            self.env_ = gym.make(temp_params["env"],max_episode_steps = (self.search_params_["horizon"]), _params=temp_params["params"])
            # self.env_ = gym.make(self.env_params_["env"],max_episode_steps = (self.search_params_["horizon"]*2), _params=self.env_params_["params"])
            self.env_.reset()

            nextNode = 0
            treePrintList = []
            done = False

            ######
            #
            #
            # Currently this does not reset at leaf nodes. It should
            #
            ########
            while not(self.tree_[nextNode].is_terminal_) and not(done) and self.m_ < self.alg_params_["search"]["max_samples"]:
                nextNode,done = self.treeStep(nextNode)
                treePrintList.append(nextNode)
            if not(self.tree_[nextNode].is_terminal_):
                rolloutReward = self.rollout(self.tree_[nextNode],self.search_params_["rollout"])
            else:
                rolloutReward = self.bounds_[1]/(1-self.alg_params_["gamma"])
            
            #rolloutReward += distance to goal()
            self.tree_[nextNode].V_ = rolloutReward
            self.backpropagate(nextNode,rolloutReward)
            # return treePrintList
            # treePrintList[0] = "Iteration" + " " + self.m_.__str__() + ": " + nodeTrajectory[0].__str__()
            # print(*nodeTrajectory, sep = " -> ")
        # Q = -np.inf
        # for a in self.tree_[0].a_:
        #     Qn = 0
        #     for r,s,n in  zip(a.r_, a.s_prime_i_,a.n_):
        #         Qn += (r + self.alg_params_["gamma"]*self.tree_[s].V_)*n 
        #     if a.N_ != 0:
        #        Qn /= a.N_
        #     if Qn > Q:
        #         Q = Qn
        #         bestAction = a
            # print("Q:", Qn, "R:", a.r_, "N:", a.N_, "n:", a.n_, "s':", a.s_prime_, "action:", a.a_)#, "V:",self.tree_[s].V_, "action:", a.a_)#, "State", s)
            # for sp in a.s_prime_i_:
            #     print ("V:", self.tree_[sp].V_, self.tree_[sp].V_/self.tree_[sp].N_)
        #bestAction = self.as_s_.return_action(self.tree_[0],[],self)
        # plt.cla()
        # for s in self.tree_:
        #     # print(s.s_)
        #     if "pose" in s.s_:
        #         self.map_[s.s_["pose"][0]][s.s_["pose"][1]] +=1
        # # print("---g")
        # t_map = (self.map_)
        # # print("max map ", np.max(np.max(self.map_)))
        # plt.imshow(np.transpose(t_map), cmap='Reds', interpolation='hanning')
        # plt.pause(1)

        return self.as_s_.return_action(self.tree_[0],[0],self)


    def select(self, nodeIndex,_param = []):
        """
        Expand tree at given state
        Args:
            self (UCT): UCT Object
            _vertex (Vertex): State or Action Vertex to expand
            _n_iter (int): Number of nodes to process in tree
        Returns:
            Action: Best Action from Current state
        """
        done = False
        action = self.as_s_.return_action(self.tree_[nodeIndex],_param,self)
        obs,reward,done = self.simulate(self.tree_[nodeIndex].s_, action)
        nextNodeIndex = self.tree_[nodeIndex].add_child(action,obs,self.n_,reward)
        if nextNodeIndex == self.n_:
            self.tree_[nextNodeIndex] = State(obs,self.env_.get_actions(obs),[nodeIndex],0)
            self.tree_[nextNodeIndex].N_ += 1 
            self.n_ += 1
        self.tree_[nextNodeIndex].is_terminal_ = done
        if done:
            self.tree_[nextNodeIndex].V_ = self.bounds_[1]/(1-self.alg_params_["gamma"])  
         #add if nextNodeIndex == self.n_: done = true to before the return
         
        return nextNodeIndex, action, done
    
    def expand(self,nodeIndex):
        action = random.choice(self.tree_[nodeIndex].a_unused)
        # print(action)
        obs,reward,done = self.simulate(self.tree_[nodeIndex].s_, action)
        nextNodeIndex = self.tree_[nodeIndex].add_child(action,obs,self.n_,reward)
        self.tree_[nextNodeIndex] = State(obs,self.env_.get_actions(obs),[nodeIndex],0)
        self.tree_[nextNodeIndex].is_terminal_ = done
        self.tree_[nextNodeIndex].N_ += 1 
        self.n_ += 1
        if done:
            self.tree_[nextNodeIndex].V_ = self.bounds_[1]/(1-self.alg_params_["gamma"])     
        return nextNodeIndex


    def treeStep(self, nextNode = 0):
        done = False
        if len(self.tree_[nextNode].a_unused) > 0 :
            nextNode = self.expand(nextNode)
            done = True
        else:
            nextNode, action, done = self.select(nextNode)
        return nextNode, done



    def simulate(self, _s, _a):
        # self.map_[_s["pose"][0]][_s["pose"][1]] += 1

        state, reward, done, info = self.env_.step(_a)
        # print(_a)

        self.m_ += 1
        return state,reward,done

    def rollout(self, _s,_n = 0,_param = None):
        """
        Performs rollout on UCT
        Args:from solvers.uct import UCT
            self (UCT): UCT Object
            _s (State): State to perform rollout at
            _a (actions): List of potential actions
            _param (dict) : Dictionary of action parameters
            _n (int): Iteration Remaining
        Returns:
            r: Estimate of reward
        """
        reward = 0
        for i in range(_n):
            a = self.as_r_.return_action(_s,_param)
            # print(a)
            s,r,done = self.simulate(_s.s_, a)                
            reward += (self.alg_params_["gamma"]**i)*r
            if done or self.m_ >= self.alg_params_["search"]["max_samples"]:
                reward += r/(1-self.alg_params_["gamma"])
                break
        return reward

    def backpropagate(self, _v_i, _val):

        # traverse up parents (using indices) call compute Q.
        i_parent : int = self.tree_[_v_i].get_parent()
        r = 0
        for i in i_parent:
            for a in self.tree_[i].a_:
                for si,s in enumerate(a.s_prime_i_):
                    if s == _v_i:
                        r = a.r_[si]
        nodeValue = self.alg_params_["gamma"]* _val + r #)/self.tree_[i_parent[0]].N_
        N = self.tree_[i_parent[0]].N_
        self.tree_[i_parent[0]].V_ = self.tree_[i_parent[0]].V_*(N-1)/N +nodeValue/N

        if i_parent[0] != 0:
            self.backpropagate(i_parent[0],self.tree_[i_parent[0]].V_)
            #print(self.tree_[i_parent[0]].V_)