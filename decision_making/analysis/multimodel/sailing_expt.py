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


params = {"mm_env": "irl_gym/MultiModel-v0", "models": [], "belief": {"type":"uniform", "params":{}}}
shared_params = {
    "max_steps": 100,
    "map_size": [30,30],
    "goal": [10,10],
    "state":{"pose": [12,12]},
    "dt": 0.11,
    "timeout_mult": 5,
    "velocity_lim": [0,2],
    # "continuity_mode": "continuous",
    "env": "irl_gym/GridWorld-v0",
    
    
    "observer_local": False,
    "render": "plot",
    "cell_size": 20,
    "prefix": current + "/plot/",
    "save_frames": False,
    "save_gif": True,
    "log_level": "WARNING"
}

model_flags = {
        "objects": True, 
        "repair": True, 
        "slip": False, 
        "omni_drive": False, 
        "battery": False, 
        "obstacles": False, 
        "obj_partially_obs": False,
        "interaction_heading" : False,
        "state_heading": True,
        "kinematics": True,
        "task": False,
    }

noise_lin = 10
model_params = [{
    "mapping": {},
    "max_time": 100,
    "dimensions": [50,50],
    "r_radius": 5,
    "r_range": [-0.01, 1],
    "p": 0,
}]*2
# model_params = [{"continuity_mode": "continuous", "mapping": {}, "model": model_flags,
#                  "controller": {"lin_gain": {"kp": 1, "ki": 0.0, "kd": 0.0, "db": 0.1}, "ang_gain": {"kp": 2, "ki": 0, "kd": 0, "db": 0.01}, "is_feedforward": False, "db_gain": 0.1},
#                  "slip": {"value": {"var":[noise_lin, noise_lin,0]}, "limits": {"var":[0, 1]}},
#                  "battery": {"value" : {"level": 100, "decay": [0.05, 0.2, 1, 2, 0, 0]}, "limits": {"level": [0, 100], "decay": [-100, 10]}},
#                 #  "obstacles": {"max_num": 15, "max_radius": 5, "max_sides": 6, "is_random": False},
#                  "resource_usage": {"value": 5.0, "limits": [0, 50], "enforce": False},
#                  "objects": {"max_num": 4, "is_random": False},
#                 #  "objects": {"objects": [{"id": 0, "pose": [20,20,0]}]},#, {"id": 1, "pose": [20,20,0]}]},
#                  "grab": {"value": {"p": 1, "is_directional": False, "taper": False}, "limits": {"range": [0, 1], "grab_radius": [0,0.5], "grab_time": [0.1, 4], "direction": [-np.pi/4, np.pi/4]}},
#                  "drop": {"value": {"p": 1, "is_directional": False, "taper": False, "drop_time": 2}, "limits": {"drop_radius": [0,0.5], "direction": [-np.pi/4, np.pi/4]}},
#                  "repair": {"value": {"stations":{"n": 2, "is_random": False}, "p": 1, "is_directional": False, "taper": False}, "limits": {"repair_radius": [0, 0.5], "repair_time": [0, 4]}},
#                 #  "repair": {"value": {"stations":[{"id": 0, "pose": [20,20,0], "repaired": 0}, {"id": 1, "pose": [30,20,0], "repaired": 0}], "p": 1, "is_directional": False, "taper": False, "repair_threshold": 0.9}, "limits": {"repair_radius": [0, 0.5], "repair_time": [0.25, 4], "direction": [-np.pi/4, np.pi/4]}},
#                  "reward": {
#                      "value":{
#                          "battery": 0, "battery_empty": -100, "time": -1, "distance": -2, "failed_grab": -60, "successful_grab": 2, "failed_drop": -20, "successful_drop": 5, "collision": 10, "repair": 5, "done": 1000, "stay": -100
#                      },
#                      "limits": {
#                          "battery": [-100,100], "time": [-100,100], "distance": [-100,100], "grab": [-100,100], "drop": [-100,100], "collision": [-100,100], "repair": [-70,100]
#                      }
#                  }
#                  }]#*2



# ground truth params
true_params = {**params, **deepcopy(shared_params), **model_params[0]}
true_env = gym.make(true_params["env"], max_episode_steps=true_params["max_steps"], params=true_params)
s, _ = true_env.reset(options = params)
print(s)
true_env.render()
# mm params
shared_params = deepcopy(shared_params)
shared_params["state"] = deepcopy(s)
shared_params["task"] = True
params = {**params, "shared": shared_params, "models": model_params}
# model_params[1] = deepcopy(model_params[0])
# model_params[1]["continuity_mode"] = "discrete"
# params["models"] = model_params

# alg paras
alg_params = {
            "alg": "mm_aags",
            "params": {
                "diffs_const": 0.1,
                "max_iter": 6000.0,
                "gamma": 1,
                "max_graph_size": 50000.0,
                "rng_seed": 45,
                "model_accuracy": {
                    "epsilon": 0.2,
                    "delta": 0.1
                },
                "model_selection": {
                    "function": "progressive_widening",
                    "params": {
                        "k": 2,
                        "a": 0.9
                    },
                    
                },
                "action_selection": {
                    "function": "mm_ambiguity_aware",
                    "params": {
                        "alpha": 0,
                        "action_prog_widening": {
                            "k": 0.5,
                            "a": 0.8
                        }
                    }
                }
            },
            "search": {
                "max_samples": 1000,
                "horizon": params["shared"]["max_steps"],
                "timeout": 5,
                "reinit": True
            }
        }

planner = get_agent(alg_params,params)

done  = False
s_prev = None
i = 0
while not done:
    a = planner.evaluate(s, alg_params["search"])
    # a={"a": 0}
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
    plt.pause(0.2)
    true_env.render()
    i += 1
