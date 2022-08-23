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


class Sailing(gym.Env):
    """
    Let's an agent traverse a world starting from 0,0
    Description:
        Agent tries to get to goal. Reward decreases from 50 to 0 in radius of 5 around goal
        Wind incurs a penalty from -1 (against wind) to 0 (in wind)
        at each timestep, the wind can stay the same or rotate 45 deg with probability p.
    User defines:
        goal, transition probability
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

    def __init__(self, _dim = [40, 40], _goal= [30, 10], _p = 0):
        """
        Constructor, initializes state
        Args:
            _p (float): transition probability 
            _goal (list(int)): coordinate of goal
        Returns:
            State: State object
        """
        super(Sailing, self).__init__()
        self.rng_ = np.random.default_rng()
        self.map_ = np.zeros(_dim)
        self.dim_ = _dim
        
        self.wind_ = np.zeros(1)
        self.resample_wind()
        self.wind_init_ = self.wind_
        # print(self.wind_)
        self.goal_ = _goal
        
        # for i in range(self.dim_[0]):
        #     for j in range(self.dim_[1]):
        #         self.map_[i][j] = self.get_reward([i,j], 0,0)
                
        self.p_ = _p
        self.reinit()
        
        self.a_ = [list(range(7))]
        
        self.fig_ = plt.figure()
        self.ax_ = self.fig_.add_subplot(1,1,1)
        
    def resample_wind(self):
        if len(self.wind_) != 1:
            for i in range(self.dim_[0]):
                for j in range(self.dim_[1]):
                    p = self.rng_.uniform()
                    if p < self.p_:
                        dir =  self.rng_.choice([-1,1])
                        dir = self.wind_[i][j] + dir
                        if dir < 0:
                            dir = 8
                        elif dir > 8:
                            dir = 0
                        self.wind_[i][j] = dir
        else:
            self.wind_ = np.zeros(self.dim_)
            for i in range(self.dim_[0]):
                for j in range(self.dim_[1]):
                    self.wind_[i][j] = self.rng_.choice(range(8))
        
    def get_num_states(self):
        return self.dim_[0]*self.dim_[1]
    
    def get_num_actions(self):
        return 8
    
    def reinit(self, _state = None):
        # print("-----------")
        # print(self.wind_)
        if _state == None:
            self.agent_ = [np.floor(self.dim_[0]/2), np.floor(self.dim_[1]/2)] 
            self.wind_ = self.wind_init_
        else:
            self.agent_ = [_state[0], _state[1]]
            self.wind_ = np.reshape(_state[2:len(_state)], [self.dim_[0], self.dim_[1]])
        # print(self.wind_)
        #self.wind_init_ = self.wind_
        
        
        
    def render(self, fp = None):
            #plt.clf()
        print(self.agent_)
        plt.cla()
        #plt.grid()
        size = 100/self.dim_[0]
        # Render the environment to the screen
        t_map = (self.map_)
        plt.imshow(np.transpose(t_map), cmap='Reds', interpolation='hanning')
        for i in range(self.dim_[0]):
                for j in range(self.dim_[1]):
                    arr = self.act_2_dir(self.wind_[i][j])
                    plt.arrow(i,j,arr[0]/3, arr[1]/3)
        if self.agent_[0] != self.goal_[0] or self.agent_[1] != self.goal_[1]:
            plt.plot(self.agent_[0], self.agent_[1],
                     'bo', markersize=size)  # blue agent
            plt.plot(self.goal_[0], self.goal_[1],
                     'gX', markersize=size)
        else:
            plt.plot(self.goal_[0], self.goal_[1],
                     'bX', markersize=size) # agent and goal

        ticks = np.arange(-0.5, self.dim_[0]-0.5, 1)
        self.ax_.set_xticks(ticks)
        self.ax_.set_yticks(ticks)
        plt.xticks(color='w')
        plt.yticks(color='w')
        plt.show(block=False)
        # if fp != None:
        #     self.fig_.savefig(fp +"%d.png" % self.img_num_)
        #     self.img_num_ += 1
        plt.pause(1)
        #plt.close() 
        
    def get_observation(self):
        return [int(self.agent_[0]), int(self.agent_[1])] + list(self.wind_.ravel())
    
    def get_distance(self, s1, s2):
        return np.sqrt( (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2 )
    
    def get_reward(self, _s, _w, _a):
        
        wind_diff = self.get_distance(self.act_2_dir(_w),self.act_2_dir(_a))
                
        # print(_w)
        # print(_a)
        # print(wind_diff/(2*np.sqrt(2)))
        d = self.get_distance(_s, self.goal_)
        if d >= 5:
            return -0.5 -wind_diff/(2*np.sqrt(2))
        else:
            return 5000*(1 - (d**2)/25) -wind_diff/(2*np.sqrt(2))
        # d = self.get_distance(_s, self.goal_)
        # return 5000*(1 - (d**2)/100) -wind_diff/(2*np.sqrt(2))
    
    def sample_transition(self, _action):
        p = self.rng_.uniform()

        if p < self.p_:
            t = self.a_.copy()
            t.remove(_action)
            _action = self.rng_.choice(t)     
        return _action
    
    def step(self, _action):
        self.map_[int(self.agent_[0])][int(self.agent_[1])]+=1

        # print("------")
        # print(self.wind_)
        wind_dir = self.wind_[int(self.agent_[0])][int(self.agent_[1])]
        # print(_action)
        # _action = self.sample_transition(_action)
        # print(_action)
        s = self.agent_.copy()
        self.agent_ = self.get_coordinate_move(self.agent_, _action)
        
        
        r = self.get_reward(s, wind_dir, _action)
        if self.agent_ == self.goal_:
            done = True
        else:
            done = False
            
        self.resample_wind()
        # print(self.wind_)
        return self.get_observation(), r, done, []
        
    def get_actions(self, _agent):
        n, a = self.get_neighbors(_agent)
        return a
    
    def get_neighbors(self, _position):
        neighbors = []
        neighbors_ind = []
        step = [[ 0, -1], [-1, -1], [-1,  0], [-1,  1], [ 0,  1], [ 1,  1], [ 1,  0], [ 1, -1]]
        for i in range(8):
            t = list(_position)
            t[0] += step[i][0]
            t[1] += step[i][1]
            
            if t[0] >= 0 and t[1] >= 0 and t[0] < self.dim_[0] and t[1] < self.dim_[1]:
                neighbors.append(t)
                neighbors_ind.append(i)
        return neighbors, neighbors_ind

    def act_2_dir(self, _action):
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
        return step
    
    def get_coordinate_move(self, _position, _action):
        _position = _position.copy()
        step = self.act_2_dir(_action)

        temp = _position.copy()
        temp[0] = temp[0] + step[0]
        temp[1] = temp[1] + step[1]
        
        if temp[0] < 0:
            temp[0] = 0
        if temp[0] >= self.dim_[0]:
            temp[0] = self.dim_[0]-1
        if temp[1] < 0:
                temp[1] = 0
        if temp[1] >= self.dim_[1]:
            temp[1] = self.dim_[1]-1
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
