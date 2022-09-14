import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import random

import gym

from gym_envs.gridworld import GridWorld
from gym_envs.gridtrap import GridTrap
from gym_envs.sailing import Sailing
from planners.aogs import AOGS
from select_action.actions import *

## Params
if False:
    fp = "/home/jared/ambiguity_ws/src/ambiguous-decision-making/python/analysis/sailing_test/"
else:
    fp = None
    
alpha = 0
timeout = 5
p = 0
test_type = 2
D = 100
# env = gym.make("GridWorld")
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

#env2 = GridWorld(dim, goal, p)


#Solver
act_select = action_selection(ambiguity_aware, [alpha])

s = env.get_observation()
aogs = AOGS(env, act_select, _performance = [0.1, 0.05], _bounds = bounds)
env.render()
r=0
d = False
max_count = 50
cnt = 0
rew = 0
while(not d and cnt < max_count):
    
    a = aogs.search(s, _D = D, _num_samples = 5000, _timeout=timeout, _reinit=False)
    print("act " + str(a))
    # print("ss ",s)
    env.reset(s)
    s, r,d,info = env.step(a)
    # print("ss ",s)
    env.render()
    #print(r)
    rew += r
    cnt+=1
if cnt < max_count:
    rew += r*(max_count-cnt)
    print(rew)
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