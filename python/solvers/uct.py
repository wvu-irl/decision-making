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
    def __init__(self, _env : gym.Env,_env_sim : gym.Env,_action_selection_select : act.action_selection, _action_selection_rollout: act.action_selection, _N : int = 10000, _c : float = 0.5, _n_rollout : int = 10*8):
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
        self.env_ = _env
        self.env_sim_ = _env_sim
        self.as_s_ = _action_selection_select
        self.as_r_ = _action_selection_rollout
    
        self.N_ : int = int(_N)
        self.c_ : float = _c
        self.n_rollout_ : int = _n_rollout
        self.tree_ : list[State] = [State([],list(range(self.env_sim_.action_space.n)),None,0) for i in range (self.N_)]
        self.n_vertices_ : int  = 1
        self.seed = np.random.randint(0,200)
        self.render_ = False
        self.terminalStates = []
        self.terminalActions = [] 
    
    def reinnit(self,obs):
        self.tree_ : list[State] = [State([],list(range(self.env_sim_.action_space.n)),None,0) for i in range (self.N_)]
        self.tree_[0].s_ = obs
        self.n_vertices_ : int  = 1

    def learn(self, s_ = 0, budget : int = 10000,gamma = 1):
        for i in range(budget):
            if i >= self.N_-1:
                break
            self.env_sim_.reset(seed = self.seed, return_info=True)
            nodeTrajectory = self.search(s_,gamma)
            nodeTrajectory[0] = "Interation" + " " + i.__str__() + ": " + nodeTrajectory[0].__str__()
            print(*nodeTrajectory, sep = " -> ")
        Q = -np.inf
        for a in self.tree_[0].a_:
            if len(a.s_prime_i_) > 0:
                for s in  a.s_prime_i_:
                    if a.r_[0] + (self.tree_[s].V_/self.tree_[s].N_) > Q:
                        Q = self.tree_[s].V_/self.tree_[s].N_
                        bestAction = a
                    print("Q:", self.tree_[s].V_/self.tree_[s].N_, "R:", a.r_, "N:", self.tree_[s].N_,"V:",self.tree_[s].V_, "action:", a.a_, "State", s)
        #bestAction = self.as_s_.return_action(self.tree_[0],[],self)
        return bestAction

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
        obs,reward,_ = self.simulate(action)
        nextNodeIndex = self.tree_[nodeIndex].add_child(action,obs,self.n_vertices_,reward)
        if reward == 20:
            self.tree_[nextNodeIndex].is_terminal_ = True
            done = True
        return nextNodeIndex, action, done
    
    def expand(self,nodeIndex):
        action = random.choice(self.tree_[nodeIndex].a_unused)
        obs,reward,_ = self.simulate(action)
        nextNodeIndex = self.tree_[nodeIndex].add_child(action,obs,self.n_vertices_,reward)
        if reward == 20:
            self.tree_[nextNodeIndex].is_terminal_ = True
        self.tree_[nextNodeIndex] = State(obs,[i for i in range(self.env_sim_.action_space.n)],[nodeIndex],0)
        self.tree_[nextNodeIndex].N_ += 1 
        self.n_vertices_ += 1
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
        while not(self.tree_[nextNode].is_terminal_) and not(done):
           nextNode,done = self.treeStep(nextNode)
           treePrintList.append(nextNode)
        if not(self.tree_[nextNode].is_terminal_):
            rolloutReward = self.rollout(self.tree_[nextNode],self.n_rollout_,gamma)
        else:
            rolloutReward = 20
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
        if self.render_:
            self.env_sim_.render()
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
            if r==20:
                return reward
        return reward

    def backpropagate(self, _v_i,reward):

        # traverse up parents (using indices) call compute Q.
        i_parent : int = self.tree_[_v_i].get_parent()
        r = 0
        for a in self.tree_[i_parent[0]].a_:
            if len(a.s_prime_i_) > 0:
                if a.s_prime_i_[0] == i_parent:
                    r = a.r_[0]
        nodeValue = (reward + r)
        self.tree_[i_parent[0]].V_ += nodeValue

        if i_parent[0] != 0:
            self.backpropagate(i_parent[0],nodeValue)
            
    def playGame(self, stateIndex = 0):
        state,_ = self.env_.reset(seed = self.seed, return_info=True)
        done = False
        self.env_.render()
        while True:
            if not(self.tree_[stateIndex].is_terminal_) and not(done):
                self.reinnit(state)
                bestAction = self.learn()
                state, reward, done, info  = self.env_.step(bestAction.a_)
                print(bestAction.a_)
                self.env_.render()
                for s in bestAction.s_prime_:
                    if self.tree_[s].s_ == state:
                        stateIndex = s
            else:
                self.env_.reset(seed = self.seed, return_info=True)
                stateIndex = 0
                done = False