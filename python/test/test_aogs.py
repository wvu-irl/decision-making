import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import random

from gym_envs.gridworld import GridWorld
from gym_envs.sailing import Sailing
from solvers.aogs import AOGS
from select_action.actions import *

## Params
alpha = 1
#Env
dim = [40,40]
goal = [10,10]
p = 0.1
env = GridWorld(dim, goal, p)


#Solver
act_select = action_selection(ambiguity_aware, [alpha])

s = env.get_observation()
aogs = AOGS(env, act_select)

while(1):
    a = aogs.search(s)
    print("act " + str(a))
    s, r,d,info = env.step(a)
    env.render()
    print(r)

#env.render("/home/jared/pomdp_ws/src/ambiguity-value-iteration/data/avi/fig")

# print("standard")
# vi = VI(opt, epsilon, gamma)
# vi.solve()
# print("DST")
# dst_vi = VI(dst_opt, epsilon, gamma)
# dst_vi.solve()

# ## Evaluate 
# r = 0
# while r != R[2]:
#     s, goal = env.get_observation()
#     a = vi.get_policy(s)
#     print(a)
#     env.step(a)
#     env.render("/home/jared/pomdp_ws/src/ambiguity-value-iteration/data/standard/fig")
#     r = env.get_reward()
#     print("reward ", r)

# env.img_num_ = 0
# env.reinit(None, seed) 
# r = 0
# while r != R[2]:
#     s, goal = env.get_observation()
#     a = dst_vi.get_policy(s)
#     print(a)
#     env.step(a)
#     env.render("/home/jared/pomdp_ws/src/ambiguity-value-iteration/data/avi/fig")
#     r = env.get_reward()
#     print("reward ", r)