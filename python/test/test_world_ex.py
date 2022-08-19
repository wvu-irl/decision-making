import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import random

from gym_envs.gridworld import GridWorld


## Params

dim = [40,40]
goal = [10,10]
p = 0.5

env = GridWorld(dim, goal, p)

while(1):
    env.step(2)
    env.render()

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