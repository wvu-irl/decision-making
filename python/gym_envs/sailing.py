import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import gym
from gym import spaces

import random
import numpy as np
import matplotlib.pyplot as plt


class AmbiguousPuddleWorld(gym.Env):
    """
    Let's an agent traverse a world with puddles where model can be presented ambiguously
    Description:
    User defines:
        _map: 2D array with 0's representing nothing and 1's representing a target object location
        _R: Vector of reward parameters [distance scale, time scale, terminal reward]
        _T: transition model for puddles. It is assumed that normal travel has p = 1 for desired transition and that wall travel has p = 1 for staying
            For puddles, assign as [ m(go), (Theta)] the mass that the robot will go as desired, stay, and the discount relative to all outcomes respectively
            As such the mass is [m(go)*(1-d), m(stay)*(1-d), d]over all outcomes respectively. m(go) + m(stay) = 1
        _seed: encodes positions of the agent and home, as well as whether or not agent has an object. The seeds also include noise for position. 
            Representation is a tuple of ten sets of 4 integers. Each four represents a value from 0-9999 which are used to interpolate between resolutions.
            For example; if the world have 10 cells in the x direction and a corresponding seed of 5501 for the agent x position, this would round to the 
                6th cell. Similarly if there were 5 cells, this would map to cell 3.
            The seed is encoded as follows:
            xxxx    xxxx    xxxx  xxxx 
            agentX  agentY  goalX goalY
    Observations:
        agent position
    Actions:
        -  S 0: move south        [ 0, -1]
        - SW 1: move southwest    [-1, -1]
        -  W 2: move west         [-1,  0]
        - NW 3: move northwest    [-1,  1]
        -  N 4: move north        [ 0,  1]
        - NE 5: move northeast    [ 1,  1]
        -  E 6: move east         [ 1,  0]
        - SE 7: move southeast    [ 1, -1]
        -  Z 8: stay              [ 0,  0]
    
    Transition: 
        movement 
    Rewards:
        - (-R_[0]*d) negative of distance
        -   R_[1]  goal reached
    Rendering:
        - blue: agent
        - green X: goal 
        - blue X: goal + agent
        - black: puddle
    """

    def __init__(self, _map, _R, _T = [0.6, 0.4, 0], _seed=None):
        """
        Constructor, initializes state
        Args:
            self (State): Object to initialize
            _x (int): x coordinate in grid world
            _y (int): y coordinate in grid world
            _battery (int): level of robot battery
            _has_obj (bool): whether or not robot has object
        Returns:
            State: State object
        """
        super(AmbiguousPuddleWorld, self).__init__()

        self.seed_length_ = 4

        self.map_ = _map
        self.map_size_ = np.shape(_map)
        self.puddle_t_ = _T
        self.B_, self.B_unambig_ = self.compute_B(_T)
        self.T_p_ = self.compute_pignistic(self.B_)
        # for x in range(self.map_size_[0]):
        #     for y in range(self.map_size_[1]):
        #         for a in range(9):
        #             print(x, " | ", y, " | ", a, " | ", self.T_p_[x][y][a])
        #         print("----------")
        self.T_unambig_p_ = self.compute_pignistic(self.B_unambig_)
        #print(self.T_unambig_p_[0][0])
        self.sample_T()
        
        # for x in range(self.map_size_[0]):
        #     for y in range(self.map_size_[1]):
        #         for a in range(9):
        #             print(x, " | ", y, " | ", a, " | ", self.T_[x][y][a])
        #         print("----------")
        # print(self.T_[0][0])
        # print(self.T_p_[5][5])
        # print(self.T_unambig_p_[5][5])
        self.R_ = _R
        self.reinit(_map, _seed)
        self.img_num_ = 0
        
        self.fig_ = plt.figure()
        self.ax_ = self.fig_.add_subplot(1,1,1)
        
        self.rng_ = np.random.default_rng()
        
    def get_num_states(self):
        return self.map_size_
    
    def get_num_actions(self):
        return 9
    
    def reinit(self, _map = None, _seed = None):
        if _seed != None:
            self.seed_ = _seed  
        else:
            self.seed_ = self.random_seed()
        
        self.agent_, self.goal_ = self.seed_2_env(self.seed_)
        self.agent_prev_ = self.agent_.copy()
        self.action_ = 9    
        
    def random_seed(self):
        seed = np.empty([self.seed_length_, 1], dtype=int)
        for i in range(self.seed_length_):
            seed[i] = random.randint(0, 9999)
        return seed

    def env_2_seed(self, _agent, _goal):
        seed = [0]* self.seed_length_

        seed[0] = np.round(_agent[0]/(self.map_size_[0]-1) * 9999)
        seed[1] = np.round(_agent[1]/(self.map_size_[1]-1) * 9999)
        seed[2] = np.round(_goal[0]/(self.map_size_[0]-1) * 9999)
        seed[3] = np.round(_goal[1]/(self.map_size_[1]-1) * 9999)

        return seed

    def seed_2_env(self, _seed):
        agent = [0,0]
        goal = [0,0]
        
        agent[0] = np.round(_seed[0]*(self.map_size_[0]-1)/9999)
        agent[1] = np.round(_seed[1]*(self.map_size_[1]-1)/9999)
        goal[0] = np.round(_seed[2]*(self.map_size_[0]-1)/9999)
        goal[1] = np.round(_seed[3]*(self.map_size_[1]-1)/9999)
        
        return agent, goal  
    
    def render(self, fp = None):
            #plt.clf()
        
        plt.cla()
        plt.grid()
        size = 100/self.map_size_[0]
        # Render the environment to the screen
        t_map = (self.map_)
        plt.imshow(np.transpose(t_map), cmap='binary', interpolation='hanning')
        if self.agent_[0] != self.goal_[0] or self.agent_[1] != self.goal_[1]:
            plt.plot(self.agent_[0], self.agent_[1],
                     'bo', markersize=size)  # blue agent
            plt.plot(self.goal_[0], self.goal_[1],
                     'gX', markersize=size)
        else:
            plt.plot(self.goal_[0], self.goal_[1],
                     'bX', markersize=size) # agent and goal

        ticks = np.arange(-0.5, self.map_size_[0]-0.5, 1)
        self.ax_.set_xticks(ticks)
        self.ax_.set_yticks(ticks)
        plt.xticks(color='w')
        plt.yticks(color='w')
        plt.show(block=False)
        if fp != None:
            self.fig_.savefig(fp +"%d.png" % self.img_num_)
            self.img_num_ += 1
        plt.pause(1)
        #plt.close() 
        
    def get_observation(self):
        return [int(self.agent_[0]), int(self.agent_[1])], [int(self.goal_[0]), int(self.goal_[1])]
    
    def get_reward(self, s1=None, a=None, s2=None):
        if s1 == None:
            s1 = self.agent_
        if s2 == None:
            s2 = self.agent_prev_
            
        if get_distance(s1,self.goal_):
            return -self.R_[0]*get_distance(s1,s2) - self.R_[1]
        else:
            return self.R_[2]
    
    def sample_distribution(self, _distribution):
        t_sum = 0
        p = self.rng_.uniform()
        # print(p)
        # print(_distribution)
        for i in range(len(_distribution)):
            t_sum += _distribution[i]
            if t_sum >= p:
                # print("i", i)
                return i
    
    def step(self, _action):
        self.agent_prev_ =self.agent_.copy()
        state_ind = self.sample_distribution(self.T_[int(self.agent_[0])][int(self.agent_[1])][_action])
        self.agent_ = self.get_coordinate_move(self.agent_, state_ind)
        
    def get_actions(self, _agent):
        n, a = self.get_neighbors(_agent)
        return a
    
    def get_neighbors(self, _position):
        neighbors = []
        neighbors_ind = []
        step = [[ 0, -1], [-1, -1], [-1,  0], [-1,  1], [ 0,  1], [ 1,  1], [ 1,  0], [ 1, -1], [0, 0]]
        for i in range(9):
            t = list(_position)
            t[0] += step[i][0]
            t[1] += step[i][1]
            
            if t[0] >= 0 and t[1] >= 0 and t[0] < self.map_size_[0] and t[1] < self.map_size_[1]:
                neighbors.append(t)
                neighbors_ind.append(i)
        return neighbors, neighbors_ind
    
    def get_transitions(self, _agent, _action):
        n, n_ind = self.get_neighbors(_agent)
        t = []
        for idx in n_ind:
            t.append(self.T_[_agent[0]][_agent[1]][_action][idx])
        return n, t 
    
    def get_transition_model(self, _agent, _action, _ambiguity = True):
        if _ambiguity:
            T = self.T_p_
        else:
            T = self.T_unambig_p_
        
            
        n, n_ind = self.get_neighbors(_agent)
        t = []
        for idx in n_ind:
            # print(_agent[0], "|", _agent[1], "|", _action, "|", idx)
            t.append(T[_agent[0]][_agent[1]][_action][idx])
        return n, t 

    def get_coordinate_move(self, _position, _action):
        _position = _position.copy()
        step = []
        if   _action == 0:
            step = [ 0, -1] # S
        elif _action == 1:
            step = [-1, -1] # SW
        elif _action == 2:
            step = [-1,  0] #  W
        elif _action == 3:
            step = [-1,  1] # NW
        elif _action == 4:
            step = [ 0,  1] # N
        elif _action == 5:
            step = [ 1,  1] # NE
        elif _action == 6:
            step = [ 1,  0] #  E
        elif _action == 7:
            step = [ 1, -1] # SE
        else:
            step = [0, 0]   #  Z

        temp = _position.copy()
        temp[0] = temp[0] + step[0]
        temp[1] = temp[1] + step[1]
        
        if temp[0] < 0:
            temp[0] = 0
        if temp[0] >= self.map_size_[0]:
            temp[0] = self.map_size_[0]-1
        if temp[1] < 0:
                temp[1] = 0
        if temp[1] >= self.map_size_[1]:
            temp[1] = self.map_size_[1]-1
        return temp

    def get_action(self, _action):
        if   _action[0] ==  0 and _action[1] == -1:
            return 0 # S
        elif _action[0] == -1 and _action[1] == -1:
            return 1 # SW
        elif _action[0] == -1 and _action[1] ==  0:
            return 2 #  W
        elif _action[0] == -1 and _action[1] ==  1:
            return 3 # NW
        elif _action[0] ==  0 and _action[1] ==  1:
            return 4 # N
        elif _action[0] ==  1 and _action[1] ==  1:
            return 5 # NE
        elif _action[0] ==  1 and _action[1] ==  0:
            return 6 #  E
        elif _action[0] ==  1 and _action[1] == -1:
            return 7 # SE
        else:
            return 8 # Z
        
    def sample_T(self):
        T = [None]*self.map_size_[0]
        for x in range(self.map_size_[0]):
            Ty = [None]*self.map_size_[1]
            for y in range(self.map_size_[1]):
                Ta = [None]*9
                for a in range(9):
                    if self.B_[x][y][a] != None:
                        # print(x, " | ", y, " | ", a, " | ", self.T_p_[x][y][a])
                        # print(self.B_[x][y][a].focal_elements) 
                        Ta[a] = self.B_[x][y][a]._sample_nonzero_probability()
                        # print(Ta[a])
                Ty[y] = Ta
            T[x] = Ty
        self.T_ = T
                
    # for bf, dempster combo is done with a list of a set of elements and a list of their corresponding masses      
    def compute_B(self, _T):
        B = [None]*self.map_size_[0]
        #print("a",np.shape(B))
        Bu = [None]*self.map_size_[0]
        sln_space = list(range(9))
        for x in range(self.map_size_[0]):
            By = [None]*self.map_size_[1]
            Buy = [None]*self.map_size_[1]
            for y in range(self.map_size_[1]):
                Ba = [None]*9
                Bau = [None]*9
                actions = self.get_actions([x,y])
                la = len(actions)
                for a in range(9):
                    bf = BeliefFunction(sln_space)
                    bfu = BeliefFunction(sln_space)                    
                    if a in actions:   
                        if la < 9:
                            bf._dempster_combination([actions],[1])
                            bfu._dempster_combination([actions],[1])
                        
                        if self.map_[x][y] == 1:
                            #fe = [[a], [8], actions]
                            if a != 8:
                                fe = [[a], [8], [a,8]]
                                m = [_T[0]*(1-_T[1]),    (1-_T[0])*(1-_T[1]),     _T[1]]
                            else:
                                fe = [[8]]
                                m = [1]
                            
                            bf._dempster_combination(fe,m)
                            fe = [[a], [8]]
                            m = [_T[0],    1-_T[0]]
                            bfu._dempster_combination(fe,m)
                            
                        else:
                            bf._dempster_combination([[a]],[1])
                            bfu._dempster_combination([[a]],[1])
                    else:
                        bf._dempster_combination([[8]],[1])
                        bfu._dempster_combination([[8]],[1])
                    Ba[a] = bf
                    Bau[a] = bfu 
                By[y] = Ba
                Buy[y] = Bau
            B[x] = By
            Bu[x] = Buy
        return B, Bu
    
    def compute_pignistic(self, B):
        T = [None]*self.map_size_[0]
        for x in range(self.map_size_[0]):
            Ty = [None]*self.map_size_[1]
            for y in range(self.map_size_[1]):
                Ta = [None]*9
                actions = self.get_actions([x,y])
                for a in range(9):#actions:              
                    Ta[a] = B[x][y][a]._get_pignistic_prob()
                Ty[y] = Ta
            T[x] = Ty
        return T