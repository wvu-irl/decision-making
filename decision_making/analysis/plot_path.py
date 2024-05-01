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

import irl_gym
import gymnasium as gym
import json
import matplotlib.pyplot as plt
import pygame

alpha = [0, 0.5, 0.75, 0.95, 1]
alpha.reverse()
# alpha = 0
# horizon = [1, 5, 10, 50, 100]
# horizon.reverse()
horizon = 10
print(alpha)
plan_params = json.load(open(current+'/../config/alg/aogs_path.json'))
env_params = json.load(open(current+'/../config/env/gt_path.json'))
env_params = json.load(open(current+'/../config/env/gt_path.json'))

paths = []
for i in range(len(alpha)):
    
    env = gym.make(env_params["env"], max_episode_steps = env_params["max_time"], params=env_params["params"])
    s,info = env.reset()
    env_params["state"] = deepcopy(s)
    plan_params["params"]["action_selection"]["params"]["alpha"] = alpha[i]
    plan_params["search"]["horizon"] = horizon#[i]

    planner = get_agent(plan_params,env_params)
    
    # print(env.get_actions(s))

    path = []
    done = False
    while(not done):
        a = planner.evaluate(s, plan_params["search"])
        s, r, done, is_trunc, info = env.step(a)
        done = done or is_trunc
        path.append(s)
        if env_params["params"]["render"] != "none":
            env.render()
            # plt.pause(0.1)
        print("Action: ", a)
    paths.append(path)
    
env = gym.make(env_params["env"], max_episode_steps = env_params["max_time"], params=env_params["params"])
env.reset()
env.render()
colors = ["purple", "red", "green", "yellow", "blue"]    
j = 0
for path in paths:
    img = env.get_image()
    print(path)
    for i in range(len(path)-1):
        pt = deepcopy(path[i]["pose"][0:2])
        pt[0] = (pt[0]+0.5)*env_params["params"]["cell_size"]
        pt[1] = (pt[1]+0.5)*env_params["params"]["cell_size"]
        pt2 = deepcopy(path[i+1]["pose"][0:2])
        pt2[0] = (pt2[0]+0.5)*env_params["params"]["cell_size"]
        pt2[1] = (pt2[1]+0.5)*env_params["params"]["cell_size"]
        # print(colors[j], pt, pt2)
        pygame.draw.line(img, colors[j], (pt[0], pt[1]), (pt2[0], pt2[1]), width=5)
    j += 1
        
    pygame.event.pump()
    pygame.display.update()
    
    pygame.image.save(img, current + "/figs/tunnel_alpha.png")
    print("Path: ", path)

    # pygame.draw.line(img, 0, (0, self._params["cell_size"] * y), (self._params["cell_size"]*self._params["dimensions"][0], self._params["cell_size"] * y), width=2)

    # done = False
    # ts = 0
    # accum_reward = 0
    # while(not done):
    #     a = planner.evaluate(s, params["algs"]["search"])
    #     s, r,done, is_trunc, info = env.step(a)
    #     done = done or is_trunc
    #     ts += 1
    #     accum_reward += r
    #     if params["envs"]["params"]["render"] != "none":
    #         env.render()
    
    # if ts < params["envs"]["max_time"]:
    #     accum_reward += (params["envs"]["max_time"]-ts)*r
    
    # data_point = nd.unstructure(params)
    # data_point["time"] = ts
    # data_point["r"] = accum_reward
    # if "pose" in data_point and "goal" in data_point:
    #     data_point["distance"] = np.linalg.norm(np.asarray(data_point["pose"][1:2])-np.asarray(data_point["goal"]))
    # data_point["final"] = deepcopy(s)
    # if "pose" in s and "goal" in data_point:
    #     data_point["final_distance"] = np.linalg.norm(np.asarray(s["pose"][1:2])-np.asarray(data_point["goal"]))
            
    # return pd.DataFrame([data_point])