#!/usr/bin/python
"""
This script is intended to run single or multithreaded decision making experiments
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from multiprocessing import Pool, Lock
import logging
import itertools
import json 
import numpy as np
from copy import deepcopy
import pickle

import gym
import irl_gym

from planners.utils import *

class RunExperiment():
    """
    Performs expermiental trials for a given algorithm and gym environment
    
    For config files, system will check "config/<algorithms, envs, or debug>/"
    
    :param alg_config: (str) Filename for algorithm params (See each alg for more details)
    :param env_config: (str) Filename for env params (see each env for more details)
    :param debug_config: (str) Filename for debug params
    
    Should be formatted as a dict of params:
    
        -
        
    :param n_trials: (int) Number of trials to run for each set of parameters, *default*: 1
    :param n_threads: (int) Number of threads to use
    """
    def __init__(self, *, alg_config : str, env_config : str, debug_config : str, n_trials : int = 1, n_threads : int = 1):
        f = open(current + "/../config/debug/" + debug_config +  ".json")
        self._debug_config = json.load(f)
        
        if "log_level" not in self._debug_config:
            self._debug_config["log_level"] = logging.WARNING
        else:
            log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
            self._debug_config["log_level"] = log_levels[self._debug_config["log_level"]]
                                             
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=self._debug_config["log_level"])
        self._log = logging.getLogger(__name__)
        
        self._log.warn("RunExperiment Init, perform " + str(n_trials) + " trials across " + str(n_threads) + "threads")
        
        f = open(current + "/../config/alg/" + alg_config +  ".json")
        self._alg_config = json.load(f)
        self._log.warn("Accessed algorithm configuration")
        f = open(current + "/../config/env/" + env_config +  ".json")
        self._env_config = json.load(f)  
        self._log.warn("Accessed Environment configuration") 
        
        self._n_trials = n_trials
        self._n_threads = n_threads  
        
        self.__lock = lock = Lock() 
        
        
    def _generate_trials(self):
        pass
    
    def _start_pool(self):
        pass
    
    def _simulate(self, params : dict):
        pass
    
    def run(self):
        self._generate_trials()
        self._start_pool()
        pass
    

## Evaluation -------------------------------------------------
def runWrapper(params : dict): 
    env = gym.make(params["env"]["env"],max_episode_steps = params["env"]["max_time"], params=params["env"]["params"])
    s = env.reset()
    params["env"]["state"] = deepcopy(s)
    planner = get_agent(params["alg"],params["env"])

    # Simulate
    print(params["trial"], params["instance"])
    # print(params)
    
    done = False
    ts = 0
    accum_reward = 0
    print(params["trial"], params["instance"], params["alg"]["search"]["max_samples"])

    while(not done):
        # print(params["trial"], params["instance"], done)
        a = planner.evaluate(s, alg_config["search"])
        s, r,done, is_trunc, info = env.step(a)
        done = done or is_trunc
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
        data[params["key"]]["init"].append(params["env"]["params"]["state"])
        data[params["key"]]["goal"].append(params["env"]["params"]["goal"])
    else:
        data[params["key"]] = {"ts": ts, "R": accum_reward}
        # print("oh")
    
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
    
    rng = np.random.default_rng()
    env_str = env_config["env"].replace("irl_gym/", "")
    fp = os.path.dirname(__file__) + "/multithread/" + alg_config["alg"] + "_" + env_str + ".pkl" #mt_config["file"] + ".npy"

    temp = []
    if alg_config["alg"] != "uct":
        temp.append(mt_config["epsilon"])
        temp.append(mt_config["delta"])
    temp.append(mt_config["horizon"])
    temp.append(mt_config["max_samples"])
    if alg_config["alg"] == "uct":
       temp.append(mt_config["c"]) 
    else:
        temp.append(mt_config["alpha"])
    if not mt_config["randomize_states"]:
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

        if alg_config["alg"] != "uct": 
            alg_config["model_accuracy"]["epsilon"] = el[0]
            alg_config["model_accuracy"]["delta"] = el[1]
            alg_config["search"]["horizon"] = el[2]
            alg_config["search"]["max_samples"] = el[3]
            # print("ms", alg_config["search"]["max_samples"])
            if alg_config["alg"] == "uct":
                alg_config["action_selection"]["params"]["c"] = el[4]
            elif alg_config["alg"] == "gbop":
                alg_config["action_selection"]["move_params"]["alpha"] = el[4]    
            else:
                alg_config["action_selection"]["params"]["alpha"] = el[4]
            if not mt_config["randomize_states"]:
                env_config["params"]["state"] = el[5]
                env_config["params"]["goal"] = el[6]
                env_config["params"]["dimensions"] = el[7]
                env_config["params"]["p"] = el[8]   
            else: 
                env_config["params"]["dimensions"] = el[5]
                env_config["params"]["p"] = el[6]
        else:
            alg_config["search"]["horizon"] = el[0]
            alg_config["search"]["max_samples"] = el[1]
            alg_config["action_selection"]["decision_params"]["c"] = el[2]
            if not mt_config["randomize_states"]:
                env_config["params"]["state"]["pose"] = el[3]
                env_config["params"]["goal"] = el[4]
                env_config["params"]["dimensions"] = el[5]
                env_config["params"]["p"] = el[6]   
            else: 
                env_config["params"]["dimensions"] = el[3]
                env_config["params"]["p"] = el[4]    
        
        # print({"env": env_config, "alg": alg_config})#, "trial": count, "instance": i})    
        for i in range(mt_config["n_trials"]):
            # print({"env": env_config.copy(), "alg": alg_config.copy(), "trial": count, "instance": i})
            if mt_config["randomize_states"]:
                s = [rng.integers(0,mt_config["world_size"][0][0]), rng.integers(0,mt_config["world_size"][0][1])]
                g = [rng.integers(0,mt_config["world_size"][0][0]), rng.integers(0,mt_config["world_size"][0][1])]
                while np.linalg.norm(np.asarray(s) -np.asarray(g)) < 5:
                    g = [rng.integers(0,mt_config["world_size"][0][0]), rng.integers(0,mt_config["world_size"][0][1])]

                env_config["params"]["state"]["pose"] = s
                env_config["params"]["goal"] = g
                if env_config["env"] == "irl_gym/Sailing-v0":
                    env_config["params"]["state"]["pose"] = [s[0], s[1], rng.integers(0,7)]
            # print(i)
            trials.append({"env": deepcopy(env_config), "alg": deepcopy(alg_config), "trial": count, "instance": i, "fp": fp, "key": data_key})
            
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
            data[el["key"]] = {"ts": [], "R": [], "init":[], "goal": []}
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
    #fp = os.path.dirname(__file__) + "/multithread/" + mt_config["file"] + ".npy"
    #with open(fp, 'wb') as f:
    #        np.save(f, data)
    # saveResultsFile(results_filename, sim_worlds)
    # if save_prev_exp:
    #     prev_exp_data.save(prev_exp_filepath)


if __name__=='__main__':
    alg_config_file = sys.argv[1]
    env_config_file = sys.argv[2]
    debug_config_file = sys.argv[3]
    if len(sys.argv >= 5):
        n_trials = sys.argv[4]
    else:
        n_trials = 1
    if len(sys.argv) > 6: 
        n_threads = int(sys.argv[5])
    else:
        n_threads = 1
        
    expts = RunExperiment(alg_config_file, env_config_file, debug_config_file, n_trials, n_threads)
    
    expts.run()

