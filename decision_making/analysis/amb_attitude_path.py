import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import csv

import numpy as np
import random

import gymnasium as gym

from decision_making.envs.grid_world import GridWorld
from decision_making.envs.grid_tunnel import GridTrap
from envs.sailing import Sailing
from planners.aogs import AOGS
from planners.uct import UCT
# from solvers.mcgs import MCGS
from decision_making.select_action import action_selection as act

import matplotlib.pyplot as plt

## functions
def compute_min_time(d):
    return np.ceil(d/(2**(1/2)))

def get_distance( s1, s2):
    return np.sqrt( (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2 )

## Params ------------------------------------------------------------------------
alg = 0
#max_samples = [100, 500, 1e3, 5e3, 1e4]
dims = [30, 35, 40, 50]
n_trials = 100
maxD = 100
test_type = 1
p = 0
timeout = 1
    
# alpha = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
alpha = [0, 0.5, 0.75, 0.95, 1]


if True:
    fp = "/home/jared/ambiguity_ws/src/ambiguous-decision-making/python/analysis/results/"
else:
    fp = None
    
file_name = "ambiguity_attitude_p" + str(p) + ".npy"
path = fp + file_name
data = []
# h = ["r_vi", "r_avi", "min_distance", "min_time", "distance_vi", "distance_avi", "time_vi",  "time_avi", "ambiguity", "probability"]
#data.append(max_samples)
n_steps = np.zeros([n_trials, len(dims), len(alpha)])
min_d = np.ones([n_trials, len(dims), len(alpha)])*500
max_d = np.zeros([n_trials, len(dims), len(alpha)])

#[0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
# timeout = 10
    
## Testing --------------------------------------------------
paths = []

for j in range(len(alpha)):
    dim = [35,10]
    goal = [10,5]
    D = 10
    env = GridTrap(dim, goal, p)
    bounds = [0,1]
    act_select = act.action_selection(act.ambiguity_aware, [alpha[j]])
    planner = AOGS(env, act_select, _performance = [0.1, 0.05], _bounds = bounds, _gamma = 0.99)
    s = env.get_observation()
        

    env.reset()
    s = env.get_observation()
    done = False
    d = 0
    temp = []
    while not done and d < maxD:
        print( "alpha", alpha[j], "depth", d)
            
        if planner.n_ > 5e4 or test_type == 2:
            do_reinit = True
        else:
            do_reinit = False
        a = planner.search(s, _D = D, _num_samples = 5000, _timeout=timeout, _reinit=do_reinit)
        # print(planner.m_)
        env.reset(s)
        s, reward ,done ,info = env.step(a)
        # print(s[0:3], reward)
        # env.get_reward(s,True)
        # env.render()
        temp.append(s)
        d+=1
    paths.append(temp)
    print(alpha[j], env.get_distance(s,env.trap_))

t_map = (env.map_)
plt.imshow(np.transpose(t_map), cmap='Reds', interpolation='hanning')            
# plt.hold(True)
for i in range(len(paths)):
    x = [25]
    y = [5]
    for el in paths[len(paths)-i-1]:
        x.append(el[0])
        y.append(el[1])

    plt.plot(x,y, linewidth=3)
labels = []
plt.xlabel("x",fontsize='xx-large')
plt.ylabel("y",fontsize='xx-large')
alpha.reverse()
for a in alpha:
        labels.append("a = " + str(a))
plt.legend(labels, loc='center left', fontsize='xx-large')    
plt.show()
plt.pause(1)

# img_num =0
# plt.savefig(fp +"figs/%d.png" % img_num)
# plt.savefig(fp +"figs/%d.eps" % img_num)

while 1:
    pass
