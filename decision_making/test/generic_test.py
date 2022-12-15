#!/usr/bin/python

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from multiprocessing import Pool, Lock

import json 
import copy
import gym
import irl_gym

from planners.utils import *

"""_summary_ This file is intended to run a user specified
        decision making algorithm from those available in config
        
    
"""
## CONFIG -------------------------------------------------
alg_config_file = sys.argv[1]
env_config_file = sys.argv[2]
if len(sys.argv) >= 4:
    max_ts = int(sys.argv[3])
else:
    max_ts = 100
# alg_config_file = "aogs"
# env_config_file = "gridworld"
# max_ts = 50
# if len(sys.argv) >= 4:
#     num_cores = sys.argv[3]
# else:
#     num_cores = 1
# if len(sys.argv) >= 5:
#     fp = sys.argv[4]
# else:
#     fp = None

f = open(current + "/../config/algorithms/" + alg_config_file +  ".json")
alg_config = json.load(f)
f = open(current + "/../config/envs/" + env_config_file +  ".json")
env_config = json.load(f)  

# SETUP CONFIG -------------------

# ENVS
# gym_examples:gym_examples/GridWorld-v0
if "r_range" in env_config:
    env_config["r_range"] = tuple(env_config["r_range"])

print("-----------")
print("-----------")
env = gym.make(env_config["env"],max_episode_steps = max_ts, params=copy.deepcopy(env_config["params"]))
print("-----------")
print("-----------")
s, info = env.reset()
env_config["state"] = copy.deepcopy(s)
# print(env_config, "--------")
print("-----------")
print("-----------")
planner = get_agent(alg_config,env_config)

# Simulate
print("-----------")

done = False

while(not done):
    print("-----------")
    print("state ",s)
    # print(done)
    a = planner.evaluate(s, alg_config["search"])#, alg_config["horizon"], alg_config["max_time"], alg_config["reinit"])
    # print("act " + str(a))
    #env.reset(s)
    # s, r,done,info = env.step(a)
    print(a)
    s, r,done, is_trunc, info = env.step(a)
    done = done or is_trunc
    env.render()
    

