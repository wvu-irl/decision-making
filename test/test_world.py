import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import random

from solver.optimization import Bellman, PignisticBellman
from solver.VI import VI

from env.AmbiguousPuddleWorld import AmbiguousPuddleWorld
from env.PuddleWorldGen import PuddleWorldGen

## Params
world_size = 10
puddle_transition = [0.9, 0.5]
R = [0, 1, 50]

map = PuddleWorldGen(world_size,world_size,0)
map.add_rectangle_puddle(3, 3, 2, 2)
map.add_rectangle_puddle(4, 4, 2, 2)
map.add_rectangle_puddle(5, 5, 2, 2)
map.add_rectangle_puddle(6, 6, 2, 2)
# map.add_rectangle_puddle(3, 3, 5, 5)
#map.add_rectangle_puddle(0,0,10,10)
epsilon = 5e1
gamma = 0.97

rng = np.random.default_rng()

## Initialize
seed = [250, 250, 9000, 9000]#np.round(rng.uniform(0,9999,4))

env = AmbiguousPuddleWorld(map.get_coarsened_world(world_size,world_size), R, puddle_transition, list(seed))

opt = Bellman(env)
dst_opt = PignisticBellman(env)

env.render("/home/jared/pomdp_ws/src/ambiguity-value-iteration/data/avi/fig")

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