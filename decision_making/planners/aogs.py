"""
The AOGS class implements in the Ambiguous Optimistic Graph Search from 

<insert citation>
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os
import pygame
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import math
import numpy as np
import time

import gymnasium as gym
import time
from copy import deepcopy

from problem.state_action import State, Action
from select_action.utils import * 


class AOGS(gym.Env):
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
        super(AOGS, self).__init__()
        if "search" in _alg_params:
            self.search_params_ = _alg_params["search"]
        _alg_params = _alg_params["params"]; 
        self.alg_params_ = _alg_params
        self.env_params_ = _env_params
        
        self.act_sel_ = action_selection.action_selection(act_sel_funcs[_alg_params["action_selection"]["function"]], _alg_params["action_selection"]["params"])
        
        self.bounds_ = [_env_params["params"]["r_range"][0]/(1-_alg_params["gamma"]), _env_params["params"]["r_range"][1]/(1-_alg_params["gamma"])]

        self.m_ = 0
        
        self.map_ =  np.zeros(self.env_params_["params"]["dimensions"])
        self._img_count = 0
        self.window = None
        self.clock = None
        
        # self._id_action = {
        #     0: np.array([ 0, -1]),
        #     1: np.array([-1, -1]),
        #     2: np.array([-1,  0]),
        #     3: np.array([-1,  1]),
        #     4: np.array([ 0,  1]),
        #     5: np.array([ 1,  1]),
        #     6: np.array([ 1, 0]),
        #     7: np.array([ 1, -1]),
        # }
        # self._triangle = [np.array([0.3,0]),np.array([-0.3,-0.15]),np.array([-0.3,0.15])]
        # for i, el in enumerate(self._triangle):
        #     self._triangle[i] = el*self.env_params_["params"]["cell_size"]
    

        # if self.env_params_["params"]["render"] == "plot":
        #     self._goal_polygon = [  (self.env_params_["params"]["goal"]+np.array([ 1  , 0.5]))*self.env_params_["params"]["cell_size"], 
        #                             (self.env_params_["params"]["goal"]+np.array([ 0.5, 1  ]))*self.env_params_["params"]["cell_size"], 
        #                             (self.env_params_["params"]["goal"]+np.array([ 0,   0.5]))*self.env_params_["params"]["cell_size"], 
        #                             (self.env_params_["params"]["goal"]+np.array([ 0.5, 0  ]))*self.env_params_["params"]["cell_size"]]
        
    
        self.reinit()
        
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
        self.env_ = gym.make(self.env_params_["env"],max_episode_steps = self.search_params_["horizon"], params=deepcopy(self.env_params_["params"]))
        
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
        
        self._state = deepcopy(_s)
        
        if _str_s not in self.gi_:
            self.gi_[_str_s] = self.n_
            a_list, neighbors = self.env_.get_actions(_s)
            self.graph_[self.n_] = State(deepcopy(_s), a_list, _L= self.bounds_[0], _U = self.bounds_[1])
            self.U_.append(_str_s) 
            self.n_ += 1
            
        self.is_not_converged_ = True
        # print('lol')
        while (time.perf_counter()-start_time < self.search_params_["timeout"]) and self.n_ < self.alg_params_["max_graph_size"] and len(self.U_) and self.m_ < self.search_params_["max_samples"] and self.is_not_converged_:
            temp_params = deepcopy(self.env_params_)
            temp_params["params"]["state"] = deepcopy(_s)
            # print('before flag ')
            # print(hash(str(_s)))
            # gym.make(self.env_params_["env"],max_episode_steps = self.search_params_["horizon"], params=self.env_params_["params"])
            # self.env_ = gym.make(self.env_params_["env"],max_episode_steps = (self.search_params_["horizon"]*2), _params=self.env_params_["params"])
            # print("s outer", _s)
            s, info = self.env_.reset(options=temp_params["params"])
            # print('after flag')
            # print(hash(str(s)))
            
            # print("sa outer", s)
            # print(len(self.U_))
            # print("------------")
            # for i in range(len(self.gi_)):
            #     print(self.graph_[i].s_)
            # print(self.gi_)    
            # if _str_s not in self.U_:
            #     # print("nee")
            #     s = self.graph_[self.gi_[self.rng_.choice(self.U_)]].s_
            # else:
                # print("yee")
            
            # print(hash(str(s)))
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
                # print("---")
                # print(s)
                # print("n " + str(self.n_) + ", d " + str(d) )
                # print("s ", s)
                str_s = hash(str(s))
                # print(str_s)
                # if str_s in self.gi_ and self.graph_[self.gi_[str_s]].s_["pose"] != s["pose"]:
                #     print("---")
                #     print(s)
                #     print(str_s)
                #     print(self.graph_[self.gi_[str_s]].s_)
                if str_s not in parents:     
                    parents[p_ind] = str_s
                    p_ind += 1
                # print(str_s)
                #pass alpha into initialization, 
                # bounds and params available from solver 
                # print('flag')
                # print(s["pose"])
                a, v_opt, L, U, diffs, exps = self.act_sel_.return_action(self.graph_[self.gi_[str_s]],[1],self)
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
                        v = r/(1-self.alg_params_["gamma"])         
                        L = v
                        U = v
                    else:
                        v = 0
                    
                    a_list, neighbors = self.env_.get_actions(s_p)
                    self.graph_[self.gi_[str_sp]] = State(deepcopy(s_p), a_list, str_s, v, is_terminal, _L = L, _U = U)
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
                    
                self.d_ += 1
                policy = self.graph_[self.gi_[str_s]].policy_
                pol_ind = self.graph_[self.gi_[str_s]].get_action_index(policy)
                # print(self.graph_[self.gi_[str_s]].s_)
                # print(self.graph_[self.gi_[str_s]].a_)
                # print(pol_ind, len(self.graph_[self.gi_[str_s]].a_))
                if str_s in self.U_ and self.graph_[self.gi_[str_s]].a_[pol_ind].N_ > self.t_:
                    self.U_.remove(str_s)
                elif str_s not in self.U_ and self.graph_[self.gi_[str_s]].a_[pol_ind].N_ <= self.t_: 
                    self.U_.append(str_s)
                    
                s = s_p
            
            parents.reverse()   
             
            self.backpropagate(list(set(parents)))
    
        # print("n " + str(self.n_))
        a, e_max, L, U, diffs, exps = self.act_sel_.return_action(self.graph_[self.gi_[_str_s]],[],self)
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
        
        
        # self.plot()
        
        
        return a#self.graph_[self.gi_[_str_s]].get_action_index(a)
               
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
            self.map_[_s["pose"][0]][_s["pose"][1]] += 1
            if _do_reset: 
                temp_params = deepcopy(self.env_params_)
                temp_params["params"]["state"] = deepcopy(_s)
                # gym.make(self.env_params_["env"],max_episode_steps = self.search_params_["horizon"], _params=self.env_params_["params"])
                # print(_s)
                s, info = self.env_.reset(options=temp_params["params"])
                # print("sa", s)
                # print(_s, self.env_.get_observation())
            # print(_a)
            # print(self.env_.step(_a))
            # exit()
            s_p, r, done, is_trunc, info = self.env_.step(_a)
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
            s_p, r = self.graph_[self.gi_[hash(str(_s))]].a_[act_ind].sample_transition_model(self.rng_)
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
                a, v, L, U, diffs, exps = self.act_sel_.return_action(self.graph_[self.gi_[s]],[],self)
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
                self.graph_[self.gi_[s]].policy_ = a
                # print("V", v)
                # print("L", L)
                # print("U", U)
                # lmn = 0
        _parents.clear()

    # def plot(self):  
    #     if self.env_params_["params"]["render"] == "plot":
    #         if self.window is None:
    #             pygame.init()
    #             pygame.display.init()
    #             self.window = pygame.display.set_mode((self.env_params_["params"]["dimensions"][0]*self.env_params_["params"]["cell_size"], self.env_params_["params"]["dimensions"][1]*self.env_params_["params"]["cell_size"]))
    #         if self.clock is None:
    #             self.clock = pygame.time.Clock()
            
    #         img = pygame.Surface((self.env_params_["params"]["dimensions"][0]*self.env_params_["params"]["cell_size"], self.env_params_["params"]["dimensions"][1]*self.env_params_["params"]["cell_size"]))
    #         img.fill((255,255,255))

    #         max_map = np.max(np.max(self.map_))
    #         for x in range(self.env_params_["params"]["dimensions"][0]):
    #             for y in range(self.env_params_["params"]["dimensions"][1]):
    #                 if self.map_[x][y] > 0:
    #                     samples = max(255*(1-self.map_[x][y]/max_map)**(4),0)
    #                     pygame.draw.rect(img, (samples,samples/2,samples/2), (x*self.env_params_["params"]["cell_size"], y*self.env_params_["params"]["cell_size"], self.env_params_["params"]["cell_size"], self.env_params_["params"]["cell_size"]))

    #         # Agent, goal
    #         if np.all(self._state["pose"] == self.env_params_["params"]["goal"]):
    #             pygame.draw.polygon(img, (255,0,0), self._goal_polygon)
    #         else:
    #             pygame.draw.circle(img, (0,0,255), (self._state["pose"]+0.5)*self.env_params_["params"]["cell_size"], self.env_params_["params"]["cell_size"]/2)
    #             pygame.draw.polygon(img, (0,255,0), self._goal_polygon)
            
    #         for y in range(self.env_params_["params"]["dimensions"][1]):
    #             pygame.draw.line(img, 0, (0, self.env_params_["params"]["cell_size"] * y), (self.env_params_["params"]["cell_size"]*self.env_params_["params"]["dimensions"][0], self.env_params_["params"]["cell_size"] * y), width=2)
    #         for x in range(self.env_params_["params"]["dimensions"][0]):
    #             pygame.draw.line(img, 0, (self.env_params_["params"]["cell_size"] * x, 0), (self.env_params_["params"]["cell_size"] * x, self.env_params_["params"]["cell_size"]*self.env_params_["params"]["dimensions"][1]), width=2)
                
    #         self.window.blit(img, img.get_rect())
    #         pygame.event.pump()
    #         pygame.display.update()
    #         self.clock.tick(4)
            
    #         if self.env_params_["params"]["save_frames"]:
    #             pygame.image.save(img, self.env_params_["params"]["prefix"] + "img" + str(self._img_count) + ".png")
    #             self._img_count += 1
                
    #     elif self.env_params_["params"]["render"] == "print":
    #         self._log.warning(str(self._state))
            
    def plot(self):
        """    
        Renders environment
        
        Has two render modes: 
        
        - *plot* uses PyGame visualization
        - *print* logs pose at Warning level

        Visualization
        
        - blue triangle: agent
        - green diamond: goal 
        - red diamond: goal + agent
        - orange triangle: wind direction
        - Grey cells: The darker the shade, the higher the reward
        """
        if self.env_params_["params"]["render"] == "plot":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.env_params_["params"]["dimensions"][0]*self.env_params_["params"]["cell_size"], self.env_params_["params"]["dimensions"][1]*self.env_params_["params"]["cell_size"]))
            if self.clock is None:
                self.clock = pygame.time.Clock()
            
            img = pygame.Surface((self.env_params_["params"]["dimensions"][0]*self.env_params_["params"]["cell_size"], self.env_params_["params"]["dimensions"][1]*self.env_params_["params"]["cell_size"]))
            img.fill((255,255,255))
            
            max_map = np.max(np.max(self.map_))
            for x in range(self.env_params_["params"]["dimensions"][0]):
                for y in range(self.env_params_["params"]["dimensions"][1]):
                    if self.map_[x][y] > 0:
                        samples = max(255*(1-self.map_[x][y]/max_map)**(4),0)
                        pygame.draw.rect(img, (samples,samples/2,samples/2), (x*self.env_params_["params"]["cell_size"], y*self.env_params_["params"]["cell_size"], self.env_params_["params"]["cell_size"], self.env_params_["params"]["cell_size"]))

            
            # Reward, wind
            for i in range(self.env_params_["params"]["dimensions"][0]):
                for j in range(self.env_params_["params"]["dimensions"][1]):                    
                    wind_direction = self._id_action[self._state["wind"][i][j]]
                    wind_direction = np.arctan2(wind_direction[1],wind_direction[0])
                    triangle = self._rotate_polygon(self._triangle,wind_direction)
                    for k, el in enumerate(triangle):
                        triangle[k] = el + (np.array([i,j])+0.5)*self.env_params_["params"]["cell_size"]
                    pygame.draw.polygon(img, (100,100,100), triangle)
            
            # Agent, goal
            if np.all(self._state["pose"] == self.env_params_["params"]["goal"]):
                pygame.draw.polygon(img, (255,0,0), self._goal_polygon)
            else:
                move_direction = self._id_action[self._state["pose"][2]]
                move_direction = np.arctan2(move_direction[1],move_direction[0])
                triangle = self._rotate_polygon(self._triangle,move_direction)
                for i, el in enumerate(triangle):
                    triangle[i] = 2.5*el + (self._state["pose"][0:2]+0.5)*self.env_params_["params"]["cell_size"]
                pygame.draw.polygon(img, (0,0,255), triangle)
                pygame.draw.polygon(img, (0,255,0), self._goal_polygon)
            
            # Grid
            for y in range(self.env_params_["params"]["dimensions"][1]):
                pygame.draw.line(img, 0, (0, self.env_params_["params"]["cell_size"] * y), (self.env_params_["params"]["cell_size"]*self.env_params_["params"]["dimensions"][0], self.env_params_["params"]["cell_size"] * y), width=2)
            for x in range(self.env_params_["params"]["dimensions"][0]):
                pygame.draw.line(img, 0, (self.env_params_["params"]["cell_size"] * x, 0), (self.env_params_["params"]["cell_size"] * x, self.env_params_["params"]["cell_size"]*self.env_params_["params"]["dimensions"][1]), width=2)
                
            self.window.blit(img, img.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(4)
            
            if self.env_params_["params"]["save_frames"]:
                pygame.image.save(img, self.env_params_["params"]["prefix"] + "img" + str(self._img_count) + ".png")
                self._img_count += 1
                
        elif self.env_params_["params"]["render"] == "print":
            p = self._state["pose"]
                
    def _rotate_polygon(self, vertices : list, angle : float, center : np.ndarray = np.zeros(2)):
        """
        Rotates a polygon by a given angle

        :param vertices: (list(ndarray)) List of 2d coordinates
        :param angle: (float) angle in radians to rotate polygon
        :param center: (ndarray) coordinate about which to rotate polygon, *default*: [0,0] 
        """
        # Since there are only a few angles, could potentially preload all of them them pass in the matrix
        vertices = deepcopy(vertices)
        R = np.zeros([2,2])
        R[0,0] = np.cos(angle)
        R[0,1] = -np.sin(angle)
        R[1,0] = np.sin(angle)
        R[1,1] = np.cos(angle)
        for i in range(len(vertices)):
            vertices[i] -= center
            vertices[i]  = np.matmul(R,vertices[i])
            vertices[i] += center
        return vertices