import sys
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


## Params ------------------------------------------------------------------------
alg = 0
max_samples = [1e3]#[100, 500, 1e3]#, 5e3]#[], 1e4]
#max_samples = [5e3]
n_trials = 50
D = 150
test_type = 2
ds = 37
    
alpha = 0
p = 0.1

if True:
    fp = "/home/jared/ambiguity_ws/src/ambiguous-decision-making/python/analysis/results/"
else:
    fp = None
    
file_name = "alg" + str(alg) + "_test" + str(test_type) + "_alpha" + str(alpha) + "_ds_" + str(ds) + ".npy"
path = fp + file_name
data = []
# h = ["r_vi", "r_avi", "min_distance", "min_time", "distance_vi", "distance_avi", "time_vi",  "time_avi", "ambiguity", "probability"]
data.append(max_samples)
r = np.zeros([n_trials,len(max_samples)])
#[0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
 ###########################################-> P = 0.2!!!
# timeout = 10

# env = gym.make("GridWorld")
if test_type == 0:
    #Env
    dim = [40,40]
    goal = [10,10]
    env = GridWorld(dim, goal, p)
    bounds = [-0.01,1]
    n_roll = 40
elif test_type == 1:
    dim = [35,10]
    goal = [10,5]
    D = 10
    env = GridTrap(dim, goal, p)
    bounds = [0,1]
else:
    #Env
    dim = [50,50]
    goal = [12,12]
    env = Sailing(dim, goal, p)
    n_roll = 20
    bounds = [-401.11, 1100.99]

s = env.get_observation()
   
if alg == 0:
    act_select = act.action_selection(act.ambiguity_aware, [alpha])
    planner = AOGS(env, act_select, _performance = [0.1, 0.05], _bounds = bounds)
elif alg == 1:
    actionSelectionSelection = act.action_selection(act.UCB1,{"c":10}) 
    actionSelectionRollout = act.action_selection(act.randomAction)

    planner = UCT(env,env.get_actions(s),actionSelectionSelection,actionSelectionRollout, _n_rollout = n_roll)
    planner.render_ = True
    planner.seed = 5
else:
    act_select_bounds = act.action_selection(act.mcgs_dm)
    act_select_move = act.action_selection(act.mcgs_best_action)
    # s = env.get_observation()
    # env.render()
    planner = MCGS(env, act_select_bounds,act_select_move, _bounds = bounds)
    
## Testing --------------------------------------------------
for i in range(len(max_samples)):
    
    for j in range(n_trials):
        env.reset()
        s = env.get_observation()
        done = False
        d = 0
        while not done and d < D:
            print("alg", alg, "test", test_type, "samples ", max_samples[i], "alpha", alpha, "trial", j, "depth", d, "ds", ds)
            
            if alg == 0:
                if planner.n_ > 5e4:
                    do_reinit = True
                else:
                    do_reinit = False
                a = planner.search(s, _D = 10, _num_samples = max_samples[i], _reinit = do_reinit)#, _timeout=timeout, _reinit=True)
            elif alg == 1:
                planner.reinit(s)
                a = planner.learn(s, _num_samples = max_samples[i])
            else:
                if planner.n_ > 5e4:
                    do_reinit = True
                else:
                    do_reinit = False
                a = planner.search(s,4,4, _H = D, _num_samples = max_samples[i])#,_timeout=timeout, _reinit=False )

            
            env.reset(s)
            s, reward ,done ,info = env.step(a)
            print(s[0:3], reward)
            
            r[j][i] += reward
            if done:
                # print(D-d)
                r[j][i] += (D-d)*reward
            d+=1
        print(r[j][i])
        with open(path, 'wb') as f:
            np.save(f, r)
            

print(r)

# with open(path, 'rb') as f:
#     np.load(f)