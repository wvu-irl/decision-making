#!/usr/bin/python

from genericpath import exists
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# from os.path import exists

from multiprocessing import Pool, Lock

import itertools
import json 
import gym
import numpy as np
import custom_gym
import copy
import pickle

from planners.utils import *

# Unrelated note: Sample MCTS graph like RRT. Then try to move twoards that subgoal, instead of normal sim. (similar to planning trees IROS paper)


"""_summary_ This file is intended to run a user specified
        decision making algorithm from those available in config
        
    
"""
lock = Lock()

## Evaluation -------------------------------------------------
def runWrapper(params : dict): 
    env = gym.make(params["env"]["env"],max_episode_steps = params["env"]["max_time"], _params=params["env"]["params"])
    planner = get_agent(params["alg"],params["env"])

    # Simulate
    print(params["trial"], params["instance"])
    # print(params)
    s = env.reset()
    done = False
    ts = 0
    accum_reward = 0
    print(params["trial"], params["instance"], params["alg"]["search"]["max_samples"])

    while(not done):
        # print(params["trial"], params["instance"], done)
        a = planner.search(s, alg_config["search"])
        s, r,done,info = env.step(a)
        ts += 1
        accum_reward += r
        
    if ts < params["env"]["max_time"]:
        # print(D-d)
        accum_reward += (params["env"]["max_time"]-ts)*r
        
    print(params["trial"], params["instance"], params["alg"]["search"]["max_samples"], s, ts, accum_reward)
    params["data"] = {"time": ts, "accum_reward": accum_reward}
    
    lock.acquire()
    with open(params["fp"], "rb") as f:
        data = pickle.load(f)
    
    if params["key"] in data:
        data[params["key"]]["ts"].append(ts)
        data[params["key"]]["R"].append(accum_reward)
    else:
        data[params["key"]] = {"ts": ts, "R": accum_reward}
        print("oh")
    
    with open(params["fp"], 'wb') as f:
        pickle.dump(data,f)      
    
    lock.release()
    # if so open
    # append ts, r
    # save
    # fp = os.path.dirname(__file__) + "/multithread/" + mt_config["file"] + ".npy"
    # with open(fp, 'wb') as f:
    #         np.save(f, data)
    # return params
    
def poolHandler(alg_config, env_config, mt_config):
    env_str = env_config["env"].replace("custom_gym/", "")
    fp = os.path.dirname(__file__) + "/multithread/" + alg_config["alg"] + "_" + env_str + ".pkl" #mt_config["file"] + ".npy"

    temp = []
    temp.append(mt_config["epsilon"])
    temp.append(mt_config["delta"])
    temp.append(mt_config["horizon"])
    temp.append(mt_config["max_samples"])
    temp.append(mt_config["alpha"])
    temp.append(mt_config["initial_state"]) # These should be provided as a bumber and sampled
    temp.append(mt_config["goal_state"])    # These should be prior sampled
    temp.append(mt_config["world_size"])
    temp.append(mt_config["probabilities"])

    temp = list(itertools.product(*temp)) 


    trials = []
    count = 0
    for el in temp:
        # print(el)
        
        data_key = ""
        for itm in el:
            data_key += str(itm) + "_"

        alg_config["model_accuracy"]["epsilon"] = el[0]
        alg_config["model_accuracy"]["delta"] = el[1]
        alg_config["search"]["horizon"] = el[2]
        alg_config["search"]["max_samples"] = el[3]
        # print("ms", alg_config["search"]["max_samples"])
        alg_config["action_selection"]["params"]["alpha"] = el[4]
        env_config["params"]["state"] = el[5]
        env_config["params"]["goal"] = el[6]
        env_config["params"]["dimensions"] = el[6]
        env_config["params"]["p"] = el[8]    
        
        # print({"env": env_config, "alg": alg_config})#, "trial": count, "instance": i})    
        for i in range(mt_config["n_trials"]):
            # print({"env": env_config.copy(), "alg": alg_config.copy(), "trial": count, "instance": i})
            trials.append({"env": copy.deepcopy(env_config), "alg": copy.deepcopy(alg_config), "trial": count, "instance": i, "fp": fp, "key": data_key})
            
        count += 1

    # print("=================")
    # for i in range(len(trials)):
    #     print("---",i,"---")
    #     print(trials[i])
    # print("fp exists", exists(fp))
    if exists(fp):
        #load
        with open(fp, "rb") as f:
            data = pickle.load(f)
        print(data)
        # exit()
    else:   
        data = {}
    print("-----", data)
        
    for el in trials:
        if el["key"] not in data:
            data[el["key"]] = {"ts": [], "R": []}
            # print("hai")
        
    with open(fp, 'wb') as f:
        pickle.dump(data,f)
        
    # Run pool of Monte Carlo trials
    print("Beginning Monte Carlo trials...")
    if mt_config["n_threads"] > 1:
        p = Pool(mt_config["n_threads"])
        # for i in range(len(trials)):
        #     print(trials[i]["trial"], trials[i]["instance"], trials[i]["alg"]["search"]["max_samples"])
        data = p.map(runWrapper, trials)
    else:
        for t in trials:
            runWrapper(t)
    print("Simulations complete.")

    # Save results
    fp = os.path.dirname(__file__) + "/multithread/" + mt_config["file"] + ".npy"
    with open(fp, 'wb') as f:
            np.save(f, data)
    # saveResultsFile(results_filename, sim_worlds)
    # if save_prev_exp:
    #     prev_exp_data.save(prev_exp_filepath)



if __name__=='__main__':
    ## CONFIG -------------------------------------------------
    alg_config_file = sys.argv[1]
    env_config_file = sys.argv[2]
    mt_config_file = sys.argv[3]

    f = open(current + "/../config/algorithms/" + alg_config_file +  ".json")
    alg_config = json.load(f)
    f = open(current + "/../config/envs/" + env_config_file +  ".json")
    env_config = json.load(f)  
    f = open(current + "/../config/multithread/" + mt_config_file +  ".json")
    mt_config = json.load(f)    

    poolHandler(alg_config, env_config, mt_config)

## Post Processing---------------------------------------------