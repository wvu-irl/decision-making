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


class GridTrap(gym.Env):
    """
    Let's an agent traverse a world starting from 0,0
    Description:
        Agent tries to get to goal. Reward decreases from 1 to 0 in radius of 5 around goal
    User defines:
        goal, transition probability
        
    Observations:
        agent position
    Actions:
        -  S 0: move south        [ 0, -1]
        -  W 2: move west         [-1,  0]
        -  N 4: move north        [ 0,  1]
        -  E 6: move east         [ 1,  0]
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

    def __init__(self, _dim = [40, 40], _goal= [10, 10], _p = 0):
        """
        Constructor, initializes state
        Args:
            _p (float): transition probability 
            _goal (list(int)): coordinate of goal
        Returns:
            State: State object
        """
        super(GridTrap, self).__init__()

        self.map_ = np.zeros(_dim)
        self.dim_ = _dim
        
        self.p_ = _p
        self.goal_ = _goal
        self.reset()
        self.trap_ = [self.agent_[0] + 2, self.agent_[1]]

        for i in range(self.dim_[0]):
            for j in range(self.dim_[1]):
                self.map_[i][j] = self.get_reward([i,j])
        
        self.a_ = [0, 1, 2, 3]
        
        self.fig_ = plt.figure()
        self.ax_ = self.fig_.add_subplot(1,1,1)
        
        self.rng_ = np.random.default_rng()
        self.count_im_ = 0
        


        
    def get_num_states(self):
        return self.dim_[0]*self.dim_[1]
    
    def get_num_actions(self):
        return 4
    
    def reset(self, _state = None):
        if _state == None:
            self.agent_ = [self.dim_[0]-self.goal_[0], self.dim_[1]-self.goal_[1]]
        else:
            self.agent_ = _state
            
        
        
        
    
    def render(self, _fp = None):
            #plt.clf()
        print(self.agent_)

        plt.cla()
        #plt.grid()
        size = 100/self.dim_[0]
        # Render the environment to the screen
        t_map = (self.map_)
        print("max map ", np.max(np.max(self.map_)))
        plt.imshow(np.transpose(t_map), cmap='Reds', interpolation='hanning')
        if self.agent_[0] != self.goal_[0] or self.agent_[1] != self.goal_[1]:
            plt.plot(self.agent_[0], self.agent_[1],
                     'bo', markersize=size)  # blue agent
            plt.plot(self.goal_[0], self.goal_[1],
                     'gX', markersize=size)
        else:
            plt.plot(self.goal_[0], self.goal_[1],
                     'bX', markersize=size) # agent and goal

        # ticks = np.arange(-0.5, self.dim_[0]-0.5, 1)
        # self.ax_.set_xticks(ticks)
        # self.ax_.set_yticks(ticks)
        plt.xticks(color='w')
        plt.yticks(color='w')
        plt.show(block=False)
        if _fp != None:
            self.fig_.savefig(_fp +"%d.png" % self.img_num_)
            self.fig_.savefig(_fp +"%d.eps" % self.img_num_)
            self.img_num_ += 1
        plt.pause(1)
        # if _fp != None:
        #     plt.savefig(_fp + "img" + str(self.count_im_) + ".png", format="png", bbox_inches="tight", pad_inches=0.05)
        #     self.count_im_+=1


        #plt.close() 
        
    def get_observation(self):
        return [int(self.agent_[0]), int(self.agent_[1])]
    
    
    def get_distance(self, s1, s2):
        return np.sqrt( (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2 )
    
    
    def get_reward(self, _s, _p = False):
        d = self.get_distance(_s, self.goal_)
        if _p:
            print(d)
        if d >= 5:
            d = self.get_distance(_s, self.trap_)
            if _p:
                print(d)
                print(_s)
                print(self.trap_)
            if d >= 5:
                return 0
            else:
                return (1 - (d**2)/25)/3
        else:
            return 1 - (d**2)/25
    
    def sample_transition(self, _action):
        p = self.rng_.uniform()

        if p < self.p_:
            # t = self.a_.copy()
            # t.remove(_action)
            # _action = self.rng_.choice(t)     
            _action = 4
        return _action
    
    def step(self, _action):
        # self.map_[int(self.agent_[0])][int(self.agent_[1])]+=1

        # print("------")
        # print(_action)
        _action = self.sample_transition(_action)
        # print(_action)
        self.agent_ = self.get_coordinate_move(self.agent_, _action)
        
        
        r = self.get_reward(self.agent_)
        if self.agent_ == self.goal_:
            done = True
        else:
            done = False
        return self.agent_, r, done, []
        
    def get_actions(self, _agent):
        n, a = self.get_neighbors(_agent)
        return a
    
    def get_neighbors(self, _position):
        neighbors = []
        neighbors_ind = []
        step = [[ 0, -1], [-1,  0], [ 0,  1], [ 1,  0]]
        for i in range(4):
            t = _position.copy()
            t[0] += step[i][0]
            t[1] += step[i][1]
            
            if t[0] >= 0 and t[1] >= 0 and t[0] < self.dim_[0] and t[1] < self.dim_[1]:
                neighbors.append(t)
                neighbors_ind.append(i)
        return neighbors, neighbors_ind

    def get_coordinate_move(self, _position, _action):
        _position = _position.copy()
        step = []
        if   _action == 0:
            step = [ 0, -1] # S
        elif _action == 1:
            step = [-1,  0] #  W
        elif _action == 2:
            step = [ 0,  1] # N
        elif _action == 3:
            step = [ 1,  0] # NE
        else:
            step = [0, 0]   #  Z

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
        # print (_position)
        # print(temp)
        return temp

    def get_action(self, _action):
        if   _action[0] ==  0 and _action[1] == -1:
            return 0 # S
        elif _action[0] == -1 and _action[1] ==  0:
            return 1 #  W
        elif _action[0] ==  0 and _action[1] ==  1:
            return 2 # N
        elif _action[0] ==  1 and _action[1] ==  0:
            return 3 #  E
        else:
            return 4 # Z
        
    def write_gif(self):
        pass