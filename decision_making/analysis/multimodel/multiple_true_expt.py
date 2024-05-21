"""
This module contains methods for generating and updating maps
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pandas as pd
import numpy as np
import nestifydict as nd
from copy import deepcopy

import gymnasium as gym

import matplotlib.pyplot as plt

from copy import deepcopy

import irl_gym

import time

from decision_making.planners.utils import get_agent


# from irl_gym.envs.structures.multimodel.mm_foraging import ForagingEnv

# generate group maps, then specify in assumptions whether this is to be used with each model.

#set of model params

def true_expt(params : dict):
    
    env_params = params["env"]
    alg_params = params["alg"]
    
    ##
    ## Model processing
    ##
    mm_params = env_params["multimodel"] #params = {"mm_env": "irl_gym/MultiModel-v0", "models": [], "belief": {"type":"uniform", "params":{}}}
    shared_params = mm_params["shared"]
    
    true_params = mm_params["true_params"]
    true_params["model"] = mm_params["flags"]
    
    true_env_params = {**deepcopy(mm_params), **deepcopy(shared_params), **deepcopy(true_params)}
    true_env = gym.make(true_env_params["env"], max_episode_steps=true_env_params["max_steps"], params=true_env_params)
    
    s, _ = true_env.reset(options = true_params)

    true_env.render()

    #because we are assuming true models. If not doing so will need flags
    shared_params["state"] = deepcopy(s)
    
    if "num_models" not in mm_params:
        mm_params["num_models"] = 1
    
    mm_params = []
    true_env_params = {**deepcopy(mm_params), **deepcopy(shared_params), **deepcopy(true_params)}
    if len(params["mm_params"]) == 0:
        for i in range(mm_params["num_models"]):
            mm_params.append(deepcopy(true_params))
        
    else:
        for i, model in enumerate(params["mm_params"]):
            temp = deepcopy(true_env_params)
            temp.update(deepcopy(mm_params[i]))
            mm_params.append(temp)

    ##
    ## Alg processing
    ##
    alg_params["search"]["horizon"] = shared_params["max_steps"]
    planner = get_agent(alg_params,mm_params)
    
    ##
    ## Data processing
    ##
    data_point = {}
    data_point["env"] = true_env_params["env"]
    data_point["max_steps"] = shared_params["max_steps"]
    data_point["map_size"] = shared_params["map_size"]
    data_point["dt"] = shared_params["dt"]
    data_point["T"] = shared_params["timeout_mult"]*shared_params["dt"]
    data_point["max_time"] = data_point["max_steps"]*data_point["T"]
    data_point["continuity_mode"] = true_params["continuity_mode"]
    if data_point["continuity_mode"] == "discrete":
        data_point["ds"] = data_point["T"]*shared_params["velocity_lim"][1]
    else:
        data_point["ds"] = None
    data_point["num_models"] = mm_params["num_models"]
    
    data_point["alpha"] = alg_params["search"]["params"]["action_selection"]["params"]["alpha"]
    
    
    
    #save dimensions
    if data_point["env"] == "irl_gym/Forging-v0":
        
        data_point = {**data_point, **true_env.get_stats()}

        
    elif data_point["env"] == "irl_gym/GridWorld-v0":
        
        #save reward
        #save number of timesteps, max_time
        #save reward (accum)

        pass
    elif data_point["env"] == "irl_gym/SailingBR-v0":
        
        #save reward
        #save number of timesteps
        #save number of reef collisions, max_time
        #save reward (accum)
        pass
    data_point["distribution"] = [] #(model, belief)
    

    ##
    ## Run Experiment
    ##

done  = False
s_prev = None
i = 0
while not done:
    a = planner.evaluate(s, alg_params["search"])
    print(a)
    s, r, done, is_trunc, _ = true_env.step(a)
    if s_prev is not None:
        ds = np.linalg.norm(np.array(s["pose"][0:2])-np.array(s_prev["pose"][0:2]))
    else:
        ds =0
    s_prev = deepcopy(s)
    print("state",s)
    print("reward",r)
    print("|||||||||||||||||||||||||||||||")
    true_env.render()
    plt.pause(1)
    i += 1

exit()

    env = gym.make(params["envs"]["env"], max_episode_steps = params["envs"]["max_time"], params=deepcopy(params["envs"]["params"]))
    s,info = env.reset()
    params["envs"]["state"] = deepcopy(s)
    planner = get_agent(params["algs"],params["envs"])

    done = False
    ts = 0
    accum_reward = 0
    min_distances = []
    min_d = np.inf
    while(not done):
        a = planner.evaluate(s, params["algs"]["search"])
        s, r,done, is_trunc, info = env.step(a)
        done = done or is_trunc
        ts += 1
        accum_reward += r
        if params["envs"]["params"]["render"] != "none":
            env.render()
        dists = [s["pose"][0], s["pose"][1], params["envs"]["params"]["dimensions"][0]-1-s["pose"][0], params["envs"]["params"]["dimensions"][1]-1-s["pose"][1]]
        min_distances.append(min(dists))
        if min(dists) < min_d:
            min_d = min(dists)
    
    if ts < params["envs"]["max_time"]:
        accum_reward += (params["envs"]["max_time"]-ts)*r
    
    data_point = nd.unstructure(params)
    data_point["time"] = ts
    data_point["r"] = accum_reward
    if "pose" in data_point and "goal" in data_point:
        data_point["distance"] = np.linalg.norm(np.asarray(data_point["pose"][1:2])-np.asarray(data_point["goal"]))
    data_point["final"] = deepcopy(s)
    if "pose" in s and "goal" in data_point:
        data_point["final_distance"] = np.linalg.norm(np.asarray(s["pose"][1:2])-np.asarray(data_point["goal"]))
    data_point["avg_min_dist"] = np.mean(min_distances)
    data_point["min_dist"] = min_d
            
    return pd.DataFrame([data_point])

import json

data = json.load(open("../config/alg/aags_temp.json"))

print(rl_expt(data))