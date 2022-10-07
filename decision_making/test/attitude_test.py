wimport sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import csv

import numpy as np
import random

import gym

from decision_making.envs.grid_world import GridWorld
from decision_making.envs.grid_tunnel import GridTrap
from envs.sailing import Sailing
from planners.aogs import AOGS
from planners.uct import UCT
from decision_making.planners.gbop import MCGS
from select_action import actions as act

## functions
def compute_min_time(d):
    return np.ceil(d/(2**(1/2)))

def get_distance( s1, s2):
    return np.sqrt( (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2 )

## Params ------------------------------------------------------------------------
alg = 1
#max_samples = [100, 500, 1e3, 5e3, 1e4]
dims = [25, 30, 35, 40, 50]
n_trials = 25
maxD = 100
test_type = 1
p = 0
timeout = 1
ds = 0
    
alpha = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]

if True:
    fp = "/home/jared/ambiguity_ws/src/ambiguous-decision-making/python/analysis/results/"
else:
    fp = None
    
file_name = "ambiguity_attitude_p" + str(p) + "_ds" + str(ds) + "mcgs.npy"
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
        # act_select = act.action_selection(act.ambiguity_aware, [alpha[j]])
        # planner = AOGS(env, act_select, _performance = [0.1, 0.05], _bounds = bounds, _gamma = 0.99)
        act_select_bounds = act.action_selection(act.mcgs_dm)
        act_select_move = act.action_selection(act.mcgs_best_action)
    # s = env.get_observation()
    # env.render()
        planner = MCGS(env, act_select_bounds,act_select_move, _bounds = bounds, _alpha = alpha[j])
    
        s = env.get_observation()
        
        for k in range(n_trials):
            env.reset()
            s = env.get_observation()
            done = False
            planner.reinit(s)
            d = 0
            while not done and d < maxD:
                print("dim", dims[i], "alpha", alpha[j], "trial", k, "depth", d)
            
                if planner.n_ > 5e4 or test_type == 2:
                    do_reinit = True
                else:
                    do_reinit = False
                a = planner.search(s,4,4, _H = D, _max_samples = 5000, _timeout=timeout, _reinit=do_reinit)

            
                env.reset(s)
                s, reward ,done ,info = env.step(a)
                print(s[0:3], reward)
                # env.render()
                n_steps[k][i][j] = d
                dist = get_distance(s,goal)
                if dist < min_d[k][i][j]:
                    min_d[k][i][j] = dist
                if dist > max_d[k][i][j]:
                    max_d[k][i][j] = dist
                d+=1
            with open(path, 'wb') as f:
                np.save(f, (n_steps, min_d, max_d))
            

# print(r)
print(n_steps)
print(min_d)
print(max_d)

with open(path, 'rb') as f:
    np.load(f)