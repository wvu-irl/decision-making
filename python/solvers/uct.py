import copy
from multiprocessing import parent_process
from platform import node
import random
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
    def __init__(self, _opt : Optimizer, _env : gym.Env,_action_selection_select : act.action_selection, _action_selection_rollout: act.action_selection, _N : int = 100000, _c : float = 0.5, _n_rollout : int = 10*8):

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

        self.opt_ = _opt
        self.env_ = _env
        self.as_s_ = _action_selection_select
        self.as_r_ = _action_selection_rollout

        self.N_ : int = int(_N)
        self.c_ : float = _c
        self.n_rollout_ : int = _n_rollout
        self.tree_ : list[State] = [State([],list(range(_env.action_space.n)),None,0) for i in range (self.N_)]
        self.current_policy : int = -1
        self.n_vertices_ : int  = 1

        self.rng_ = np.random.default_rng()
        self.render_ = False
        self.policy = None



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
        action : Action = self.as_s_.return_action(self.tree_[nodeIndex],_param,self)
        obs,reward,done = self.simulate(action)
        nextNodeIndex = self.tree_[nodeIndex].add_child(action,obs,self.n_vertices_,reward)
        self.tree_[nextNodeIndex].is_terminal_ = done
        return nextNodeIndex, action, done
    
    def expand(self,nodeIndex):
        action = random.choice(self.tree_[nodeIndex].a_unused)
        obs,reward,done = self.simulate(action)
        nextNodeIndex = self.tree_[nodeIndex].add_child(action,obs,self.n_vertices_,reward)
        self.tree_[nextNodeIndex].is_terminal_ = done
        self.tree_[nextNodeIndex] = State(obs,[i for i in range(self.env_.action_space.n)],[nodeIndex],0)
        self.tree_[nextNodeIndex].N_ += 1 
        self.n_vertices_ += 1
        return nextNodeIndex


    def search(self, s_ = 0, budget : int = 10000):
        
        """
        Conducts tree search from root
        Args:
            self (UCT): UCT Object
            _s (State): State to start search from
        Returns:
            Action: Best Action from Current state
        """
        for _ in range(budget):
            nextNode = s_
            self.env_.reset(seed=5, return_info=True)
            treePrintList = []
            while not(self.tree_[nextNode].is_terminal_):
                treePrintList.append(nextNode)
                if len(self.tree_[nextNode].a_unused) > 0 :
                    nextNode = self.expand(nextNode)
                    treePrintList.append(nextNode)
                    break
                else:
                    nextNode, action, done = self.select(nextNode)
                
            rolloutReward = self.rollout(self.tree_[nextNode],self.n_rollout_)
            self.backpropagate(nextNode,rolloutReward)
            print("Interation",_)
            print(*treePrintList, sep=" -> ")

    def simulate(self, _a):
        state, reward, done, info = self.env_.step(_a)
        if self.render_:
            self.env_.render()
        return state,reward,done

    def rollout(self, _s,_n = -1, _param = None):
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
        if _n > -1:
            a = self.as_r_.return_action(_s,_param)
            s,r,done = self.simulate(a)
            if not(done) or _n == 0:
                return r + .6*self.rollout(_s,_n-1,_param)
            else:
                return r
        return 0

    def backpropagate(self, _v_i,reward):

        # traverse up parents (using indices) call compute Q.
        i_parent : int = self.tree_[_v_i].get_parent()
        nodeValue = (reward - self.tree_[i_parent[0]].V_)/self.tree_[i_parent[0]].N_
        self.tree_[i_parent[0]].V_ += nodeValue

        if i_parent[0] != 0:
        #     i_parent_2 = self.tree_[i_parent[0]].get_parent()
        #     for a in self.tree_[i_parent_2[0]].a_:
        #         if len(a.s_prime_i_ ) > 0:
        #             if a.s_prime_i_[0] == i_parent_2 :
        #                 reward = a.r_[0]
            self.backpropagate(i_parent[0],nodeValue)
            
    def playGame(self, stateIndex = 0):
        self.render_ = True
        
        self.env_.reset(seed=5, return_info=True)
        self.env_.render()
        while True:
            if self.tree_[stateIndex].N_ != 1:
                bestValue = -np.inf
                for a in self.tree_[stateIndex].a_:
                    for s in a.s_prime_i_:
                        if self.tree_[s].V_ >= bestValue: 
                            bestValue = self.tree_[s].V_
                            bestAction = a.a_
                            stateIndex = s
                print(bestValue)
                obs =  self.env_.step(bestAction)
                self.env_.render()
            else:
                return