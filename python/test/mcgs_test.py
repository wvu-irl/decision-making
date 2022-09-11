import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import random

from gym_envs.gridworld import GridWorld
from gym_envs.gridtrap import GridTrap
from gym_envs.sailing import Sailing
from solvers.mcgs import MCGS
from select_action.actions import *

## Params
alpha = 0
#Env
dim = [25,25]
goal = [5,5]
p = 0
test_type = 1
if test_type == 0:
    #Env
    dim = [40,40]
    goal = [10,10]
    env = GridWorld(dim, goal, p)
    bounds = [0,1]
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
    bounds = [-401.11, 1100.99]

import numpy as np
#env2 = GridWorld(dim, goal, p)
timeout = 2

#Solver
act_select_bounds = action_selection(mcgs_dm)
act_select_move = action_selection(mcgs_best_action)
s = env.get_observation()
env.render()

mcgs = MCGS(env, act_select_bounds,act_select_move, _bounds = bounds)

done = False
while(not done):
    a = mcgs.search(s,4,4, _timeout=timeout, _reinit=False )
    print("act " + str(a))
    mcgs.env_.reset(s)
    s, r , done, info = env.step(a)
    env.render()
    print(r)

