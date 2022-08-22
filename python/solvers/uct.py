import copy
from multiprocessing import parent_process
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
    def __init__(self, _opt : Optimizer, _env : gym.Env,_action_selection_select : act.action_selection, _action_selection_rollout: act.action_selection, _N : int = 10000, _c : float = 0.5, _n_rollout : int = 10):

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
        self.tree_ : list[State] = [State([],[Action(i) for i in range(_env.action_space.n)]) for i in range (self.N_)]
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
        done = False
        action : Action = self.as_s_.return_action(self.tree_[nodeIndex],_param,self)
        obs,reward,_ = self.simulate(action)
        nextNodeIndex = -1
        nextNodeIndex = self.tree_[nodeIndex].add_child(action,obs,self.n_vertices_,reward)
        if nextNodeIndex == self.n_vertices_:
            self.n_vertices_ = self.n_vertices_ + 1
            self.tree_[nextNodeIndex] = State(obs,[Action(i) for i in range(self.env_.action_space.n)],[nodeIndex])
        if self.tree_[nextNodeIndex].N_ == 0:
            self.tree_[nextNodeIndex].N_ += 1
            done = True
        return nextNodeIndex, action, done
           
    def search(self, s_ = 0, budget : int = 5000):
        
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
            self.env_.reset(seed=42, return_info=True)
            while True:
                nextNode, action, fullyExpanded = self.select(nextNode)
                if fullyExpanded:
                    break
            self.tree_[nextNode].V_ = self.rollout(self.tree_[nextNode],self.n_rollout_)
            print(self.tree_[nextNode].V_)
            self.backpropagate(nextNode)

    def simulate(self, _a):
        if self.render_:
            self.env_.render()
        state, reward, done, info = self.env_.step(_a)
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
                return r + self.rollout(_s,_n-1,_param)
            else:
                return r
        return 0

    def backpropagate(self, _v_i):

        # traverse up parents (using indices) call compute Q.
        i_prime : int = self.tree_[_v_i].get_parent()
        self.tree_[i_prime[0]].V_ = self.tree_[_v_i].V_ + .9*self.tree_[i_prime[0]].V_
        if i_prime[0] != 0:
            self.backpropagate(i_prime[0])
            
    def playGame(self, stateIndex = 0):
        self.render_ = True
        
        self.env_.reset(seed=42, return_info=True) 
        while True:
            if self.tree_[stateIndex].N_ != 1:
                bestValue = -np.inf
                for a in self.tree_[stateIndex].a_:
                    for s in a.s_prime_i_:
                        if self.tree_[s].V_ > bestValue: 
                            bestValue = self.tree_[s].V_
                            bestAction = a.a_
                            stateIndex = s
                self.env_.render()
                self.env_.step(bestAction)

            else:
                return