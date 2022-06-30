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


## TODO:
# High priority -----------------------
# Write get_policy - it should evaluate Qmax over the actions
# write test MDP
# 
#
# Low priority -----------------------
# Finish reinit pruning
#
#
# Low-low priority -----------------------
# if we want, reduce size of tree to match N in evaluate policy
# 
#
#

class Vertex():
    """
    MCTS Vertex 
    Description: Vertex of MCTS tree
    User defines:
    """
    def __init__(self, _parent=-1, _state=-1, _actions = None, _child=[]):

        """
         Constructor, initializes BMF-AST
         Args:
             self (MF_KWIK_AST): Object to initialize
             gym_env (GridWorldEnv): Grid Word object for simulation
         Returns:
             BMF_AST: BMF-AST world object
         """
        super(Vertex, self).__init__()

        self.parent_ = _parent
        self.state = _state
        self.children_ = _child
        self.actions_ = []
        self.s_prime_ = []
        self.N_ = 0
        self.Q_ = 0
        self.R_ = 0


class MCTS():
    """
    Perform Monte Carlo Tree Search 
    Description: User specifies MDP model and MCTS solves the MDP policy to some confidence.
    """
    def __init__(self, _mdp, _N = 5e3, _c = 0.5, _n_rollout=50):

        """
         Constructor, initializes BMF-AST
         Args:
             self (MCTS): MCTS Object
             _mdp (MDP): Markov Decision Process to solve
             _N (int): Number of nodes to process in tree
             _c (double): exploration/exploitation term (0: total Exploitation, 1: Mostly Exploration)
             _n_rollout (int): Maximum number of rollout iterations
         Returns:
             MCTS: MCTS object
         """
        super(MCTS, self).__init__()

        self.mdp_ = _mdp

        self.N_ = int(_N)
        self.c_ = _c
        self.n_rollout_ = _n_rollout

        self.tree_ = [Vertex()] * self.N_
        self.current_policy = -1
        self.n_vertices_ = 0

        self.rng_ = np.random.default_rng()


    def reinit(self, _state, _action = None, _s_prime = None):
        """
        Reinitialize Tree from state-action-state transition
        Args:
            self (MCTS): MCTS Object
            _state (State): State Transition
            _action (Action): Action to selected
             
        Returns:
        """
        if _action == None:
             self.tree_ = [Vertex()] * self.N_
             v = Vertex(_state, -1, -1)
             self.tree_[0] = v
             self.n_vertices_ = 1
        else:
            pass
            #prune from state to action node
            #prune from action to state to state node
                #should be recursive

    def evaluate_policy(self, _N = None):
        
        """
        Evaluate Policy from Current State
        Args:
            self (MCTS): MCTS Object
            _n (int): Max number of iterations
        Returns:
            Action: Best Action from Current state
        """
        # if _N != None:
        #     if _N > self.N_:
        #         #self.tree_ = self.tree[0:_N]
        #         #Update children list of parents and remove elements from tree....
        #     self.N_ = _N

        # n = len(self.tree_)
        
        # self.tree_.extend([Vertex()] * (self.N_-n))        
        n = 0
        while n < self.N_:
            v_index = 0
            is_leaf = False
            while(not self.mdp_.is_terminal(self.tree_[v_index].state) and not is_leaf):
                v_index, is_leaf = self.select(v_index)
            self.backpropagate(v_index, self.rollout(self.tree_[v_index]))
        
        self.current_policy = self.get_policy()
        return self.current_policy
    
    def get_policy(self):
        unique_a = []
        temp_q = []
        v = self.tree_[0]
        for i in range(len(v.children_)):
            print(i)
            print(v.actions_)
            print(v.children_)

            if not (v.actions_[i] in unique_a):
                unique_a.append(v.actions_[i])
                temp_q.append(0)
            else:
                temp_q[unique_a.index(v.actions_[i])] += v.actions_[i].Q_
        
        return unique_a[temp_q.index(max(temp_q))]


    def select(self, _v_i):
        """
        Expand tree at given state
        Args:
            self (MCTS): MCTS Object
            _vertex (Vertex): State or Action Vertex to expand
            _n_iter (int): Number of nodes to process in tree
        Returns:
            Action: Best Action from Current state
        """
        if self.fully_expanded(self, _v_i):
            a = self.get_action_UCB1(_v_i)
            s = self.mdp_.transition(self.tree[_v_i].state)
            _v_i = self.get_child(_v_i, a, s)
            is_leaf = False
        else:
            _v_i = self.expand(_v_i)
            is_leaf = True

        return _v_i, is_leaf  

    def fully_expanded(self, _v_i):
        sa = self.mdp_.get_viable_transitions(self.tree[_v_i].state)
        if len(sa) == len(self.tree[_v_i].children_):
            return True
        else:
            return False
        pass

    def get_action_UCB1(self, _v_i):
        unique_actions = self.get_unique(self.tree_[_v_i].actions_)
        best_action = unique_actions[0]
        q = [0]*len(unique_actions) 
        for a in unique_actions:
            for child in range(len(self.tree_[_v_i].children)):
                a_ind = [i for i in range(len(unique_actions)) if self.tree_[child].actions_[i] == a]
                #q[a_ind] = q of child in the tree
                q[a_ind] += self.tree_[child].Q_
        max_q = max(q)
        return unique_actions.index(max_q)

    def get_unique(self, _l):
        l = []
        for item in _l:
            if item not in l:
                l.append(item)
        return l
        

    def get_child(self, _v_i, _a, _s):
        for i in range(len(self.tree_[_v_i].children_)):
            if self.tree_[_v_i].actions[i] == _a and self.tree_[_v_i].s_prime_[i] == _s:
                return i
        return -1

    def add_child(self, _v_i, _a, _s):
        self.tree_[_v_i].actions_.append(_a)
        self.tree_[_v_i].s_prime_.append(_s)
        self.n_vertices_ += 1
        self.tree_[_v_i].children_.append(self.n_vertices_)
        v = Vertex(_v_i, _s)
        self.tree_[self.n_vertices_] = v


    def expand(self, _v_i):
        (a, s) = self.get_random_action_state(_v_i)
        self.add_child(_v_i, a ,s)
        return self.n_vertices_

    def rollout(self, _s, _n = -1):
        """
        Performs rollout on MCTS
        Args:
            self (MCTS): MCTS Object
            _s (State): State to perform rollout at
            _n (int): Iteration Remaining
        Returns:
            r: Estimate of reward
        """
        if _n != -1 or self.mdp_.is_terminal(_s):
            a = self.mdp_.get_random_action()
            _s_prime = self.mdp_.transition(_s,a)
            r = self.mdp_.get_reward(_s, a, _s_prime)
            return r + self.rollout(_s, _n-1)
        else:
            return self.mdp_.approximate_value(_s)

    def backpropagate(self, _v_i, _q):
        self.tree_[_v_i].N_ += 1
        self.tree_[_v_i].Q_ += _q
        if _v_i != 0:
            self.backpropagate(self.tree_[_v_i].parent_, _q)
    
    def prune(self, _v_i, _v_child_i = -1):
        """
        Recursively prunes nodes below tree.
        Args:
            self (MCTS): MCTS Object
            _v_i (int): Index to start at
            _v_child_i (int): Index of child to prune to 
             
        Returns:
        """
        if _v_child_i == -1:
            self.decrement_children(_v_i)
        else:
            for i in range(len(self.tree_[_v_i].children)):
                curr_child_ind = self.tree_[_v_i].children[i]
                if curr_child_ind != _v_child_i:
                    for child in self.tree_[curr_child_ind].children:
                        self.prune(curr_child_ind,child)

        self.tree_pop(_v_i)
    
    def decrement_children(self, _index):
        """
        Iteratively updates children indices.
        Args:
            self (MCTS): MCTS Object
            _index (int): Index to start updating pointers
             
        Returns:
        """
        for i in range(_index+1,len(self.tree_)):
            for j in range(len(self.tree_[i].children)):
                 self.tree_[i].children[j] -= 1   

    def get_random_action_state(self, _v_i):
        viable_actions_states = self.mdp_.get_viable_transitions(self.tree_[_v_i].state)
        actions_states_children_removed = viable_actions_states.copy()
        num_children = len(self.tree_[_v_i].children_)
        for action_state in viable_actions_states:
            a = action_state[0]
            s = action_state[1]
            #if action_state in list of children of v, delete
            for i in range(num_children):
                if a == self.tree_[_v_i].actions_[i] and s == self.tree_[_v_i].s_prime_[i]:
                    actions_states_children_removed.remove(action_state)
        sampled_action_state = self.rng_.choice(actions_states_children_removed)
        return sampled_action_state