import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import csv

import numpy as np
import random

import gym

from gym_envs.gridworld import GridWorld
from gym_envs.gridtrap import GridTrap
from gym_envs.sailing import Sailing
from solvers.aogs import AOGS
from solvers.uct import UCT
# from solvers.mcgs import MCGS
from select_action import actions as act

## functions
def compute_min_time(d):
    return np.ceil(d/(2**(1/2)))

def get_distance( s1, s2):
    return np.sqrt( (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2 )

## Params ------------------------------------------------------------------------
alg = 0
#max_samples = [100, 500, 1e3, 5e3, 1e4]
dims = [30, 35, 40, 45, 50]
n_trials = 200
maxD = 100
test_type = 1
p = 0
timeout = 2
    
alpha = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]

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
for i in range(len(dims)):
    for j in range(len(alpha)):
        dim = [dims[i],10]
        goal = [10,5]
        D = 10
        env = GridTrap(dim, goal, p)
        bounds = [0,1]
        act_select = act.action_selection(act.ambiguity_aware, [alpha[j]])
        planner = AOGS(env, act_select, _performance = [0.1, 0.05], _bounds = bounds)
        s = env.get_observation()
        
    
        for k in range(n_trials):
            env.reset()
            s = env.get_observation()
            done = False
            d = 0
            while not done and d < maxD:
                print("dim", dims[i], "alpha", alpha[j], "trial", k, "depth", d)
            
                if planner.N_ > 5e4 or test_type == 2:
                    do_reinit = True
                else:
                    do_reinit = False
                a = planner.search(s, _D = D, _num_samples = 5000, _timeout=timeout, _reinit=do_reinit)

            
                env.reset(s)
                s, reward ,done ,info = env.step(a)
                print(s[0:3], reward)
            
                n_steps[k][i][j] = d
                dist = get_distance(s,goal)
                if dist < min_d[k][i][j]:
                    min_d[k][i][j] = dist
                if dist > max_d[k][i][j]:
                    max_d[k][i][j] = dist
                d+=1
            with open(path, 'wb') as f:
                np.save(f, n_steps)
                np.save(f, min_d)
                np.save(f, max_d)
            

print(r)

with open(path, 'wb') as f:
    np.load(f)