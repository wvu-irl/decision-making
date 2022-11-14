#!/usr/bin/python

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from multiprocessing import Pool, Lock

import itertools
import json 
import gym
import numpy as np
import custom_gym

from planners.utils import *

# Unrelated note: Sample MCTS graph like RRT. Then try to move twoards that subgoal, instead of normal sim. (similar to planning trees IROS paper)


"""_summary_ This file is intended to run a user specified
        decision making algorithm from those available in config
        
    
"""
   

## Evaluation -------------------------------------------------
def runWrapper(params : dict): 
    env = gym.make(params["env"]["env"],max_episode_steps = params["env"]["max_time"], _params=params["env"]["params"])
    planner = get_agent(params["alg"],params["env"])

    # Simulate
    print(params["trial"], params["instance"])
    s = env.reset()
    done = False
    ts = 0
    accum_reward = 0
    while(not done):
        # print(params["trial"], params["instance"], done)
        a = planner.search(s, alg_config["search"])
        s, r,done,info = env.step(a)
        ts += 1
        accum_reward += r
        # print(params["trial"], params["instance"], done)
    params["data"] = {"time": ts, "accum_reward": accum_reward}
    return params

        # if save_prev_exp:
        #     lock.acquire()
        #     prev_exp_data.record(obj, t)
        #     lock.release()

        # terminal_condition = obj.simulationStep(t)
        # if slow_mode and num_threads == 1:
        #     time.sleep(0.5)
        
        # if ((t == num_time_steps - 1) or (enable_terminal_condition and terminal_condition)):
        #     # Save final step of prev exp if at final time step
        #     if save_prev_exp:
        #         lock.acquire()
        #         prev_exp_data.record(obj, t)
        #         lock.release()

        #     # Display final map if at final time step
        #     if config.enable_plots and num_threads == 1:
        #         displayMap(obj, plt, map_fig, map_ax)
        #         if save_plots == 1:
        #             t = t+1
        #             map_fig.savefig("figures/fig%d.png" % t)

        # End simulation early if terminal condition reached
        # if enable_terminal_condition and terminal_condition:
        #     break
    
def poolHandler(alg_config, env_config, mt_config):

    temp = []
    temp.append(mt_config["epsilon"])
    temp.append(mt_config["delta"])
    temp.append(mt_config["horizon"])
    temp.append(mt_config["alpha"])
    temp.append(mt_config["initial_state"]) # These should be provided as a bumber and sampled
    temp.append(mt_config["goal_state"])    # These should be prior sampled
    temp.append(mt_config["world_size"])
    temp.append(mt_config["probabilities"])

    temp = list(itertools.product(*temp)) 


    trials = []
    count = 0
    for el in temp:
        alg_config["model_accuracy"]["epsilon"] = el[0]
        alg_config["model_accuracy"]["delta"] = el[1]
        alg_config["search"]["horizon"] = el[2]
        alg_config["action_selection"]["params"]["alpha"] = el[3]
        env_config["params"]["state"] = el[4]
        env_config["params"]["goal"] = el[5]
        env_config["params"]["dimensions"] = el[6]
        env_config["params"]["p"] = el[7]        
        
        for i in range(mt_config["n_trials"]):
            trials.append({"env": env_config, "alg": alg_config, "trial": count, "instance": i})
        count += 1

    
    # Save this as a csv file.  or np file

    # Run pool of Monte Carlo trials
    print("Beginning Monte Carlo trials...")
    if mt_config["n_threads"] > 1:
        p = Pool(mt_config["n_threads"])
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