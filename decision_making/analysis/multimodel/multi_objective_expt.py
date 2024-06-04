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
    
    env_params = params["envs"]
    alg_params = params["algs"]
    
    ##
    ## Model processing
    ##
    multimodel_params = env_params["multimodel"] #params = {"mm_env": "irl_gym/MultiModel-v0", "models": [], "belief": {"type":"uniform", "params":{}}}
    shared_params = env_params["shared"]
    
    true_params = env_params["true_params"]
    true_params["model"] = env_params["flags"]
    
    true_env_params = {**deepcopy(true_params), **deepcopy(shared_params)}
    true_env = gym.make(true_env_params["env"], max_episode_steps=true_env_params["max_steps"], params=true_env_params)
    
    s, _ = true_env.reset(options = true_params)

    if true_env_params["render"] != "none":
        true_env.render()
    print(s)
    
    #because we are assuming true models. If not doing so will need flags
    shared_params["state"] = deepcopy(s)
    print(shared_params["state"])
    true_params["model"]["is_truth"] = False
    
    if "num_models" not in multimodel_params:
        multimodel_params["num_models"] = 1
    else:
        multimodel_params["num_models"] = int(multimodel_params["num_models"][0])
    
    mm_params = []
    test_model_params = env_params["mm_params"]
    if len(env_params["mm_params"]) == 0:
        for i in range(multimodel_params["num_models"]):
            mm_params.append(deepcopy(true_params))
        
    else:
        for i, model in enumerate(env_params["mm_params"]):
            temp = deepcopy(true_env_params)
            temp.update(deepcopy(test_model_params[i]))
            mm_params.append(temp)

    test_env_params = {**multimodel_params, "shared" :shared_params, "models": mm_params}

    ##
    ## Alg processing
    ##
    # alg_params["search"]["horizon"] = shared_params["max_steps"]
    planner = get_agent(alg_params,test_env_params)
    
    ##
    ## Data processing
    ##
    data_point = {}
    data_point["env"] = true_env_params["env"]
    print("env",data_point["env"])
    data_point["max_steps"] = shared_params["max_steps"]
    data_point["horizon"] = alg_params["search"]["horizon"]
    data_point["max_samples"] = alg_params["search"]["max_samples"]
    data_point["search_timeout"] = alg_params["search"]["timeout"]
    data_point["action_a_param"] = alg_params["params"]["action_selection"]["params"]["action_prog_widening"]["a2"]
    data_point["model_a_param"] = alg_params["params"]["model_selection"]["params"]["a1"]
    data_point["map_size"] = shared_params["map_size"]
    data_point["dt"] = shared_params["dt"]
    data_point["T"] = shared_params["timeout_mult"]*shared_params["dt"]
    data_point["max_time"] = data_point["max_steps"]*data_point["T"]
    data_point["continuity_mode"] = true_params["continuity_mode"]
    if data_point["continuity_mode"] == "discrete":
        data_point["ds"] = data_point["T"]*shared_params["velocity_lim"][1]
    else:
        data_point["ds"] = None
    data_point["num_models"] = multimodel_params["num_models"]
    
    data_point["alpha"] = alg_params["params"]["action_selection"]["params"]["alpha"]


    ##
    ## Run Experiment
    ##

    done  = False
    is_trunc = False
    s_prev = None
    i = 0
    
    planning_time = []
    while not done and not is_trunc:
        s["is_refresh"] = True
        start_time = time.perf_counter()
        a = planner.evaluate(s, alg_params["search"])
        planning_time.append(time.perf_counter()-start_time)
        print("ACTION ------", a)
        s, r, done, is_trunc, _ = true_env.step(a)
        # print()
        if s_prev is not None:
            ds = np.linalg.norm(np.array(s["pose"][0:2])-np.array(s_prev["pose"][0:2]))
        else:
            ds =0
        s_prev = deepcopy(s)
        print(true_env.get_objects())
        print("state",s)
        print("reward",r)
        print("|||||||||||||||||||||||||||||||")
        if true_env_params["render"] != "none":
            true_env.render()
            plt.pause(1)
        
        
        if data_point["env"] == "irl_gym/Foraging-v0":            
            data_point = {**data_point, **true_env.get_stats()}
            # print("yes")

            
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
        print(data_point["time"])
        
        belief = planner.env_.get_belief()
        data_point["distribution"] = [] #(model, belief)
        for el in belief:
            data_point["distribution"].append((el, belief[el]))
            
        i += 1
        print(i, done, is_trunc)
        data_point["iteration_time"] = i
        
    data_point["planning_time"] = np.mean(planning_time)
        
    return pd.DataFrame([data_point])

# import json

# data = json.load(open("test_config/TEST_multi_true_foraging.json"))

# dp = true_expt(data)


    