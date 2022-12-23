#!/usr/bin/python
"""
This script is intended to run single or multithreaded decision making experiments
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os

from jinja2 import DictLoader
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
    
    For config files, system will check "config/<algorithms, envs, or expt>/"
    
    :param alg_config: (str) Filename for algorithm params (See each alg for more details)
        Config file should contain an element "algs" which has both "default" params common to all algs, as well as remaining as well as specific members with all possible values
    :param env_config: (str) Filename for env params (see each env for more details)
        Config file should contain an element "envs" which has both "default" params common to all envs, as well as remaining as well as specific members with all possible values
    :param n_trials: (int) Number of trials to run for each set of parameters, *default*: 1
    :param n_threads: (int) Number of threads to use, *default*: 1
    :param log_level: (str) Log level (does not override default values), *default*: WARNING
    """
    def __init__(self, *, alg_config : str, env_config : str, n_trials : int = 1, n_threads : int = 1, log_level = "WARNING"):
        
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        self._log_level = log_levels[log_level]
                                             
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=self._expt_config["log_level"])
        self._log = logging.getLogger(__name__)
        
        self._log.warn("RunExperiment Init, perform " + str(n_trials) + " trials across " + str(n_threads) + "threads")
        
        f = open(current + "/../config/alg/" + alg_config +  ".json")
        self._alg_config = json.load(f)
        self._alg_default = self._alg_config.pop('default', None)
        if "log_level" not in self._alg_default:
            self._alg_default["log_level"] = log_level
        self._log.warn("Accessed algorithm configuration")
        
        f = open(current + "/../config/env/" + env_config +  ".json")
        self._env_config = json.load(f)  
        self._env_default = self._env_config.pop('default', None)
        if "log_level" not in self._env_default:
            self._env_default["log_level"] = log_level
        self._log.warn("Accessed Environment configuration") 
        
        self._n_trials = n_trials
        self._n_threads = n_threads  
        
        self.__lock = lock = Lock() 
        
        
    def _generate_trials(self):
        """
        Generates a set of trials from the environment and algorithm params provided

        :return: (list(dict)) contains list of the parameters for each algorithm
        """
        # Generate algorithm params
        algs = [] 
        for el in self._alg_config["algs"]:
            algs.append(self.__expand_trials(el))
            
        # Generate Environment params
        envs = []
        for el in self._env_config["envs"]:
            algs.append(self.__expand_trials(el))
        
        # Combine experiments
        expts = []
        temp = list(itertools.product(*[algs,envs]))  
        for i in self._n_trials:
            expts.append(deepcopy(temp))
        
        return expts
    
    def __expand_trials(self, d, form):
        """
        Unpacks a dictionary into a sequence of lists for trial generation
        
        :param d: (dict) params to expand
        :return: (list(dict)) trials
        """
        trials = []
        # temp = list(itertools.product(*configs))
        # configs = self.__pack_dict(keys, configs, "alg") 
        # print(list(itertools.product(*a.values())))
        
        # to repack values, iterate over original dict outline then recursively call using the subkeys
        # so something like
        
        #for el in list
            #make new val        
            #for itm in el?
                #val[el] = update_dict(el,itm)
                
        #def update_dict(el,val)
            # for key, el in dict:
                # if type(el) == dict:
                    #update_dict(el, val)
                # else
                    # return el
        return deepcopy(trials)
    
    def __unpack_dict(self, d):
        """
        Converts nested dictionary into a list of lists
        
        :param d: (dict) dictionary to be mapped to a list
        :return: (list(str), list(list)) keys and values for each element in d
        """
        if type(d) == dict:
            temp_key = []
            temp_el = []
            for key in d:
                temp_key.append(key)
                keys, vals = self.__unpack_dict(d[key])
                if keys != None:
                    temp_key.append(keys)  
                temp_el.append(vals) 
            return temp_key, temp_el            
        elif type(d) == list:
            return None, d
        else:
            return None, [d]    
        
    
    def __pack_dict(self, d_in, d_structure):
        """
        Repackages list of keys and values into a nested dictionary (consumes input dictionary)

        :param d_in: 1d dict containing values
        :param d_structure: (dict) dictionary containing structure and default values
        :return: Restructured dictionary
        """
        d_out = deepcopy(d_structure)
        for key in d_structure:
            if type(d_structure[key]) == dict:
                d_out[key] = self.__pack_dict(d_in, d_structure[key])
            elif key in d_in:
                d_out[key] = deepcopy(d_in[key])
                d_in.pop(key)

        return deepcopy(d_out)
    
    def _start_pool(self, expts):
        pass
    
    def _simulate(self, params : dict):
        pass
    
    def run(self):
        expts = self._generate_trials()
        self._start_pool(expts)
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
    expt_config_file = sys.argv[3]
    if len(sys.argv >= 5):
        n_trials = sys.argv[4]
    else:
        n_trials = 1
    if len(sys.argv) > 6: 
        n_threads = int(sys.argv[5])
    else:
        n_threads = 1
        
    expts = RunExperiment(alg_config_file, env_config_file, expt_config_file, n_trials, n_threads)
    
    expts.run()

