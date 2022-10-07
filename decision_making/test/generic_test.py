#!/usr/bin/python

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from multiprocessing import Pool, Lock

import json 
import gym
import custom_gym

from planners import *
from select_action.actions import *
# from envs import *


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
if env_config["env"] == "gridworld":
    env = gym.make("custom_gym/GridWorld-v0",env_config["dimensions"], env_config["goal"], env_config["probability"])
    # env = gridworld.GridWorld(env_config["dimensions"], env_config["goal"], env_config["probability"])
elif env_config["env"] == "sailing":
    env = gym.make("custom_gym/Sailing-v0",env_config["dimensions"], env_config["goal"], env_config["probability"])
elif env_config["env"] == "gridtrap":
    env = gym.make("custom_gym/GridTunnel-v0",env_config["dimensions"], env_config["goal"], env_config["probability"])

# ALGS
if alg_config["alg"] == "aogs":
    act_sel = action_selection(ambiguity_aware, [alg_config["ambiguity_attitude"]])
    planner = aogs.AOGS(env, act_sel, alg_config["max_iter"], env_config["reward_bounds"], [alg_config["epsilon"], alg_config["delta"]], alg_config["gamma"])
    #aogs = AOGS(env, act_select, _performance = [0.1, 0.05], _bounds = bounds)

elif alg_config["alg"] == "gbop":
    #planner = gbop.GBOP(env, act_select, _performance = [0.1, 0.05], _bounds = bounds)
    pass
elif alg_config["alg"] == "uct":
    #planner = uct.UCT(env, act_select, _performance = [0.1, 0.05], _bounds = bounds)
    pass

# Simulate
s = env.reset()

ts = 0
done = False
while(not done and ts < max_ts):
    print("-----------")
    print("state ",s)
    a = planner.search(s, alg_config["max_samples"], alg_config["horizon"], alg_config["max_time"], alg_config["reinit"])
    print("act " + str(a))
    
    env.reset_step(s)
    s, r,done,info = env.step(a)
    env.render()
    ts += 1

print("state ",s)

