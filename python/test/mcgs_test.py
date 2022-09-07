import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import random

from gym_envs.gridworld import GridWorld
from gym_envs.sailing import Sailing
from solvers.mcgs import MCGS
from select_action.actions import *

## Params
alpha = 0
#Env
dim = [30,30]
goal = [25,10]
p = 0.1
sailing_test = False
if not sailing_test:
    env = GridWorld(dim, goal, p)
    bounds = [0,1]
else:
    env = Sailing(dim, goal, p)
    bounds = [-1, 5000]

#env2 = GridWorld(dim, goal, p)
timeout = 10

#Solver
act_select_bounds = action_selection(mcgs_dm)
act_select_move = action_selection(mcgs_best_action)
s = env.get_observation()
env.render()

mcgs = MCGS(env, act_select_bounds,act_select_move, _bounds = bounds)

done = False
while(not done):
    a = mcgs.search(s,4,4, _timeout=timeout, _reinit=True)
    print("act " + str(a))
    env.reinit(s)
    s, r , done, info = env.step(a)
    env.render()
    print(r)

