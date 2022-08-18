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
    def __init__(self, _opt : Optimizer, _env : gym.Env,_action_selection_select : act.action_selection, _action_selection_rollout: act.action_selection, _N : int = 5e3, _c : float = 0.5, _n_rollout : int = 50):

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

        self.tree_ : list[State] = [State([])] * self.N_
        self.current_policy : int = -1
        self.n_vertices_ : int  = 0

        self.rng_ = np.random.default_rng()
        self.render_ = False

    def reinit(self, _state : State):
        """
        Reinitialize Tree from state-action-state transition
        Args:
            self (UCT): UCT Object
            _state (State): State Transition
            _action (Action): Action to selected
             
        Returns:
        """

        #Potentially have Initializer  call reinit
        # reinit  envsocet
            #  self.tree_ = [Vertex()] * self.N_
            #  v = Vertex(_state, -1, -1)
            #  self.tree_[0] = v
            #  self.n_vertices_ = 1

    def select(self, nodeIndex,_a,_param):
        """
        Expand tree at given state
        Args:
            self (UCT): UCT Object
            _vertex (Vertex): State or Action Vertex to expand
            _n_iter (int): Number of nodes to process in tree
        Returns:
            Action: Best Action from Current state
        """
        action = self.as_s_.return_action(self.tree_[nodeIndex],_a,_param)
        state,reward,done = self.simulate(action)
        nextNodeIndex = self.tree_[nodeIndex].add_child(state,self.n_vertices_,reward)
        if  nextNodeIndex > self.n_vertices_:
            self.n_vertices_ += 1
        return nextNodeIndex, action,  (done or self.tree_[nextNodeIndex].N_ == 0)
           
    def search(self, _s : State = None, budget : int = 5000):
        
        """
        Conducts tree search from root
        Args:
            self (UCT): UCT Object
            _s (State): State to start search from
        Returns:
            Action: Best Action from Current state
        """
        for _ in range(budget):
            nextNode = self._s
            while True:
              nextNode, action, fullyExpanded = self.select(nextNode)
              if fullyExpanded:
                break
            self.tree_[nextNode].V_ = self.rollout(nextNode,action,self.n_rollout_)
            self.backpropagate(nextNode)

    def simulate(self, _a):
        if self.render_:
            self.env_.render()
        state, reward, done, info = self.env_.step(_a)
        return state,reward,done

    def rollout(self, _s, _a,_param,_n = -1):
        """
        Performs rollout on UCT
        Args:
            self (UCT): UCT Object
            _s (State): State to perform rollout at
            _a (actions): List of potential actions
            _param (dict) : Dictionary of action parameters
            _n (int): Iteration Remaining
        Returns:
            r: Estimate of reward
        """
        if _n > -1:
            a = self.as_r_.return_action(_s,_a,_param)
            s,r,done = self.simulate(a)
            if ~done:
                return r + self.rollout(_s,_a,_param,_n-1)
            else:
                return r
        return 0

    def backpropagate(self, _v_i):

        # traverse up parents (using indices) call compute Q.
        self.tree_[_v_i].N_ += 1
        self.tree_[_v_i].Q_ += self.opt.compute_Q(self.tree[_v_i])
        if _v_i != 0:
            self.backpropagate(self.tree_[_v_i].parent_)
            
    # def get_policy(self, _s : State = None, _opt : Optimizer = Bellman()):
    #     #find s in tree
    #     return self.T_(ind).policy_