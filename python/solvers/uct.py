import copy
from email.mime import base
from multiprocessing import parent_process
from platform import node
import random
from unittest.case import _BaseTestCaseContext
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from optimizers.optimization import Optimizer, Bellman
from select_action import actions as act
from problem.state_action import State, Action
import gym

class UCT():
    """
    Perform Upper Confidence Tree Search 
    Description: User specifies problem and UCT solves for the policy
    """
    def __init__(self, _env_sim : gym.Env,actions : int ,_action_selection_select : act.action_selection, _action_selection_rollout: act.action_selection, _N : int = 10000, _c : float = 0.5, _n_rollout : int = 15, _bounds = [0,1]):
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
        # self.env_ = _env
        self.env_sim_ = _env_sim
        self.actions_ = actions
        self.as_s_ = _action_selection_select
        self.as_r_ = _action_selection_rollout
        
        self.m_ = 0
        self.bounds_ = _bounds
    
        self.N_ : int = int(_N)
        self.c_ : float = _c
        self.n_rollout_ : int = _n_rollout
        self.tree_ : list[State] = [State([],self.actions_,None,0) for i in range (self.N_)]
        self.n_vertices_ : int  = 1
        self.seed = np.random.randint(0,200)
        self.render_ = False
        self.terminalStates = []
        self.terminalActions = [] 
    
    def reinit(self,obs):
        self.tree_ : list[State] = [State([],self.actions_,None,0) for i in range (self.N_)]
        self.n_vertices_ : int  = 1

    def learn(self, s_ , _num_samples = 5e3, budget : int = 5000,gamma = .9):
        self.m_ = 0
        self.n_samples_ = _num_samples
        self.gamma_ = gamma
        for i in range(budget):
            if i >= self.N_-1 or self.m_ >= self.n_samples_:
                break
            self.env_sim_.reset(_state = s_)#, seed = self.seed) #_state = s_
            nodeTrajectory = self.search(0,gamma)
            nodeTrajectory[0] = "Iteration" + " " + i.__str__() + ": " + nodeTrajectory[0].__str__()
            # print(*nodeTrajectory, sep = " -> ")
        Q = -np.inf
        for a in self.tree_[0].a_:
            Qn = 0
            for r,s,n in  zip(a.r_, a.s_prime_i_,a.n_):
                Qn += (r + self.gamma_*self.tree_[s].V_)*n 
            if a.N_ != 0:
               Qn /= a.N_
            if Qn > Q:
                Q = Qn
                bestAction = a
            # print("Q:", Qn, "R:", a.r_, "N:", a.N_, "n:", a.n_, "s':", a.s_prime_, "action:", a.a_)#, "V:",self.tree_[s].V_, "action:", a.a_)#, "State", s)
            # for sp in a.s_prime_i_:
            #     print ("V:", self.tree_[sp].V_, self.tree_[sp].V_/self.tree_[sp].N_)
        #bestAction = self.as_s_.return_action(self.tree_[0],[],self)

        return bestAction.a_

    def select(self, nodeIndex,_param = None):
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
        obs,reward,done = self.simulate(action)
        nextNodeIndex = self.tree_[nodeIndex].add_child(action,obs,self.n_vertices_,reward)
        if nextNodeIndex == self.n_vertices_:
            self.tree_[nextNodeIndex] = State(obs,self.actions_,[nodeIndex],0)
            self.tree_[nextNodeIndex].N_ += 1 
            self.n_vertices_ += 1
        self.tree_[nextNodeIndex].is_terminal_ = done
        if done:
            self.tree_[nextNodeIndex].V_ = self.bounds_[1]/(1-self.gamma_)           
        return nextNodeIndex, action, done
    
    def expand(self,nodeIndex):
        # print(self.actions_)
        action = random.choice(self.tree_[nodeIndex].a_unused)
        obs,reward,done = self.simulate(action)
        nextNodeIndex = self.tree_[nodeIndex].add_child(action,obs,self.n_vertices_,reward)
        self.tree_[nextNodeIndex] = State(obs,self.actions_,[nodeIndex],0)
        self.tree_[nextNodeIndex].is_terminal_ = done
        self.tree_[nextNodeIndex].N_ += 1 
        self.n_vertices_ += 1
        if done:
            self.tree_[nextNodeIndex].V_ = self.bounds_[1]/(1-self.gamma_)     
        return nextNodeIndex


    def search(self, baseNode = 0, gamma = .9):
        
        """
        Conducts tree search from root
        Args:
            self (UCT): UCT Object
            _s (State): State to start search from
        Returns:
            Action: Best Action from Current state
        """
        nextNode = baseNode
        treePrintList = []
        done = False
        while not(self.tree_[nextNode].is_terminal_) and not(done) and self.m_ < self.n_samples_:
           nextNode,done = self.treeStep(nextNode)
           treePrintList.append(nextNode)
        if not(self.tree_[nextNode].is_terminal_):
            rolloutReward = self.rollout(self.tree_[nextNode],self.n_rollout_,gamma)
        else:
            rolloutReward = self.bounds_[1]/(1-self.gamma_)
            
        #rolloutReward += distance to goal()
        self.tree_[nextNode].V_ = rolloutReward
        self.backpropagate(nextNode,rolloutReward)
        return treePrintList

    def treeStep(self, nextNode = 0):
        done = False
        if len(self.tree_[nextNode].a_unused) > 0 :
            nextNode = self.expand(nextNode)
            done = True
        else:
            nextNode, action, done = self.select(nextNode)
        return nextNode, done



    def simulate(self, _a):
        state, reward, done, info = self.env_sim_.step(_a)
        self.m_ += 1
        return state,reward,done

    def rollout(self, _s,_n = 0,gamma=.9,_param = None):
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
            s,r,done = self.simulate(a)                
            reward += (gamma**i)*r
            if done or self.m_ >= self.n_samples_:
                reward += r/(1-gamma)
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
        nodeValue = self.gamma_* _val + r #)/self.tree_[i_parent[0]].N_
        N = self.tree_[i_parent[0]].N_
        self.tree_[i_parent[0]].V_ = self.tree_[i_parent[0]].V_*(N-1)/N +nodeValue/N

        if i_parent[0] != 0:
            self.backpropagate(i_parent[0],self.tree_[i_parent[0]].V_)
            #print(self.tree_[i_parent[0]].V_)
            
    def playGame(self, stateIndex = 0):
        state = self.env_.reset(seed = self.seed)
        self.tree_[0].s_ = state
        done = False
        self.env_.render()
        while True:
            if not(self.tree_[stateIndex].is_terminal_) and not(done):
                self.reinit(state)
                bestAction = self.learn(state)
                state, reward, done, info  = self.env_.step(bestAction.a_)
                print(bestAction.a_)
                self.env_.render()
                # for s in bestAction.s_prime_i_:
                #     if self.tree_[s].s_[0] == state:
                #         stateIndex = s
            else:
                self.env_.reset(seed = self.seed)
                stateIndex = 0
                done = False