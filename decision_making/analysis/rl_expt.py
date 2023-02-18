import gym
import pandas as pd
import numpy as np
import nestifydict as nd
from copy import deepcopy

def rl_expt(params : dict):
    """
    Simulates and saves a single experimental trial
    
    :param params: (dict) Contains "alg" and "env" with corresponding params
    """
    print(params)
    env = gym.make(params["envs"]["env"], max_episode_steps = params["envs"]["max_time"], params=params["envs"]["params"])
    s = env.reset()
    params["envs"]["state"] = deepcopy(s)
    planner = get_agent(params["algs"]["params"],params["envs"]["params"])

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
    if "pose" in data_point and "goal" in data_point:
        data_point["distance"] = np.linalg.norm(np.asarray(data_point["pose"])-np.asarray(data_point["goal"]))
    data_point["final"] = deepcopy(s)
    if "pose" in s and "goal" in data_point:
        data_point["final_distance"] = np.linalg.norm(np.asarray(s["pose"])-np.asarray(data_point["goal"]))
            
    return pd.DataFrame([data_point])