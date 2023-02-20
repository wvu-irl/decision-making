import pandas as pd
import numpy as np
import nestifydict as nd
from copy import deepcopy

import sys
import os

current = os.path.dirname(__file__)
parent = os.path.dirname(current)
sys.path.append(parent)

from planners.utils import get_agent

def rl_expt(params : dict):
    import irl_gym
    import gymnasium as gym
    """
    Simulates and saves a single experimental trial
    
    :param params: (dict) Contains "alg" and "env" with corresponding params
    """
    print(params["envs"]["env"])
    env = gym.make(params["envs"]["env"], max_episode_steps = params["envs"]["max_time"], params=deepcopy(params["envs"]["params"]))
    s,info = env.reset()
    print(s)
    params["envs"]["state"] = deepcopy(s)
    planner = get_agent(params["algs"],params["envs"])

    done = False
    ts = 0
    accum_reward = 0
    while(not done):
        a = planner.evaluate(s, params["algs"]["search"])
        s, r,done, is_trunc, info = env.step(a)
        done = done or is_trunc
        ts += 1
        accum_reward += r
        if params["envs"]["params"]["render"] != "none":
            env.render()
    
    if ts < params["envs"]["max_time"]:
        accum_reward += (params["envs"]["max_time"]-ts)*r
    
    data_point = nd.unstructure(params)
    data_point["time"] = ts
    data_point["r"] = accum_reward
    print(data_point["pose"])
    print(data_point["goal"])
    if "pose" in data_point and "goal" in data_point:
        data_point["distance"] = np.linalg.norm(np.asarray(data_point["pose"][1:2])-np.asarray(data_point["goal"]))
    data_point["final"] = deepcopy(s)
    if "pose" in s and "goal" in data_point:
        data_point["final_distance"] = np.linalg.norm(np.asarray(s["pose"][1:2])-np.asarray(data_point["goal"]))
            
    return pd.DataFrame([data_point])