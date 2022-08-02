import copy
from multiprocessing import parent_process
import random
import numpy as np
import matplotlib.pyplot as plt


from problem.state_action import State, Action
from optimizers.optimization import Optimizer
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import gym


class MCGS():
    """
    Perform Monte Carlo Tree Search 
    Description: User specifies MDP model and MCTS solves the MDP policy to some confidence.
    """
    def __init__(self, _env : gym.Env, _opt : Optimizer, _action_selection_select, _N = 5e3,_bounds = None):

        """
         Constructor, initializes BMF-AST
         Args:
             self (MCTS): MCTS Object
             _mdp (MDP): Markov Decision Process to solve
             _N (int): Number of nodes to process in tree
             _n_rollout (int): Maximum number of rollout iterations
         Returns:
             MCTS: MCTS object
         """
        super(MCGS, self).__init__()

        self.env_ = _env
        self.bounds_ = _bounds
        self.N_ = int(_N)
        self.opt_ = _opt
        self.budget_ = 10000

        self.as_s_ = _action_selection_select
        
        #### COME BACK ######
        a = _env.action_space #Action Space
        self.a_ = a 
        self._select_param = []
        #####################

        self.graph_ : dict = {}
        for i in range(_N):
            self.graph_[i] = State()

        
        self.current_policy = -1
        self.n_vertices_ = 0

        self.rng_ = np.random.default_rng()

    ############## COME BACK ##############################
    def reinit(self, _state, _action = None, _s_prime = None):
        """
        Reinitialize Graph from state-action-state transition
        Args:
            self (MCGS): MCGS Object
            _state (State): State Transition
            _action (Action): Action to selected
             
        Returns:
        """
        if _action == None:
            self.graph_ : dict = {}
            for i in range(self.N_):
                self.graph_[i] = State()
    ######################################################
              
    def search(self, _s : State, _h :int = 100):
        """
        Conducts Graph search from root
        Args:
            self (MCGS): MCGS Object
            _s (State): State to start search from
            _h (int) : Horizon to evaluate

        Returns:

        """
        if _h > self.budget_:
            print("Please Enter a Horizon (_h) that is less than the computational Budget")
        else:
            for m in range(self.budget_):
                self.env_.reset()
                s = _s
                for t in range(_h):
                    n = (m-1)*_h + t
                    self.compute_bounds(n)
                    b = self.select(s,n)
                    obs, reward, done = self.simulate(b)
                    self.graph_eval(obs,reward)
                    if done:
                        break
               
    def compute_bounds(self, _n):
        return

    def select(self,_s):
        """
        Select action to take for MCGS Simulation
        Args:
            self (MCGS): MCGS Object
            _s (State): Current State
        Returns:
            Action: Best Action from Current state
        """
        return self.as_s_.return_action(_s,self.a_,self._select_param)

    def simulate(self, _a):
        """
        Simulate the MCGS object's Enviroment
        Args:
            self (MCGS): MCGS Object
            _a (State): Action to take
        Returns:
            obs: Observation of simulation
            r: reward collected from simulation
            done (bool): Flag for simulation completion
        """
        obs, r, done, info = self.env_.step(_a)
        return obs, r, done

    def graph_eval(self, _obs, _r):
        ## IF _Obs is in Graph Already
        ## - Add state, state_ trans, action, reward into  S and new S' node

        ## ELSE
        ##  - Add state, state_ trans, action, reward into  S and S'
        return