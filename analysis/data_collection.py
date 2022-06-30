import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import csv

import numpy as np
import random

from solver.optimization import Bellman, PignisticBellman
from solver.VI import VI

from env.AmbiguousPuddleWorld import AmbiguousPuddleWorld
from env.PuddleWorldGen import PuddleWorldGen
from utils.utils import *

## functions
def compute_min_time(d):
    return np.ceil(d/(2**(1/2)))

## Params
prefix = "/home/jared/pomdp_ws/src/ambiguity-value-iteration/data/"
attempt_num = 7

world_size = 10
puddle_transition = [0.6, 0.5]
R = [0, 1, 50]

file_name = "w" + str(world_size) + "avi" + str(attempt_num) + ".csv"
path = prefix + file_name
data = []
h = ["r_vi", "r_avi", "min_distance", "min_time", "distance_vi", "distance_avi", "time_vi",  "time_avi", "ambiguity", "probability"]
data.append(h)

map = PuddleWorldGen(world_size,world_size,0)

epsilon = 5e1
gamma = 0.97
t_max = 250

rng = np.random.default_rng()

num_goals = 5
num_init = 5
num_trials = 10
num_box_scenarios = 5
num_ambiguity = 6
num_probability = 5
max_num_boxes = 5
max_size_boxes = 5

## Initialize
num_iter = num_goals*num_init*num_trials*num_box_scenarios*num_ambiguity*num_probability
reward = np.zeros([num_iter,2])
min_distance = np.zeros([num_iter,1])
min_time = np.zeros([num_iter,1])
distance = np.zeros([num_iter,2])
time = np.zeros([num_iter,2])
ambiguity = np.zeros([num_iter,1])
probability = np.zeros([num_iter,1])

prob = list(range(1,6))
amb = list(range(6))


seed = np.round(rng.uniform(0,9999,4))
print(seed)
env = AmbiguousPuddleWorld(map.get_coarsened_world(world_size,world_size), R, puddle_transition, list(seed))

## Collect Data
k = 0
for i_goal in range(num_goals):
    seed[2] = np.round(rng.uniform(0,9999))
    seed[3] = np.round(rng.uniform(0,9999))
    for i_box in range(num_box_scenarios):
        num_boxes = int(np.ceil(rng.uniform(0,max_num_boxes)))
        map = PuddleWorldGen(world_size,world_size,0)
        for j in range(num_boxes):
            box_x = int(np.floor(rng.uniform(0,max_size_boxes)))
            box_y = int(np.floor(rng.uniform(0,max_size_boxes)))
            ag_x = int(np.floor(rng.uniform(0,world_size-box_x-2)))
            ag_y = int(np.floor(rng.uniform(0,world_size-box_y-2)))
            map.add_rectangle_puddle(ag_x, ag_y, box_x, box_y )
            
        for p in prob:
            for a in amb:
                puddle_transition = [p/5,a/5]
            
                env = AmbiguousPuddleWorld(map.get_coarsened_world(world_size,world_size), R, puddle_transition, list(seed))
                opt = Bellman(env)
                dst_opt = PignisticBellman(env)
                #env.render()
                vi = VI(opt, epsilon, gamma)
                vi.solve()
                dst_vi = VI(dst_opt, epsilon, gamma)
                dst_vi.solve()
        
                for i_init in range(num_init):
                    d = 0
                    while d < 2500:
                        seed[0] = np.round(rng.uniform(0,9999))
                        seed[1] = np.round(rng.uniform(0,9999))
                        d = get_distance([seed[0],seed[1]], [seed[2], seed[3]])
            
                    for i_trial in range(num_trials):
                        print("Goal ", i_goal, " | Box ", i_box, " | P ", p, " | A ", a, " | Init ", i_init, " | Trial ", i_trial)
                        env.sample_T()
                        
                        env.reinit(map,list(seed))
                        
                        ag, goal = env.get_observation()
                        min_distance[k][0] = get_distance(ag, goal)
                        print(ag, goal, min_distance[k][0])
                        min_time[k][0] = compute_min_time(min_distance[k][0])
                        ambiguity[k][0] = a
                        probability[k][0] = p
                        
                        print("vi")
                        env.reinit(map,list(seed))
                        t = 0
                        r = 0
                        d = 0
                        ag, g_temp = env.get_observation()
                        ag_prev = ag
                        while r != R[2] and t < 250:
                            
                            
                            act = vi.get_policy(ag)
                            #print(ag, act)
                            env.step(act)
                            # print("--------------")
                            # print(ag,goal)
                            ag, g_temp = env.get_observation()
                            r = env.get_reward()
                            reward[k][0] += r
                            t += 1
                            d += get_distance(ag, ag_prev)
                            # print(d)
                            # print("--------------")
                            # env.render()
                            ag_prev = ag
                            
                        distance[k][0] = d
                        time[k][0] = t
                        
                        print("avi")
                        env.reinit(map,list(seed))
                        t = 0
                        r = 0
                        d = 0
                        ag, g_temp = env.get_observation()
                        ag_prev = ag
                        while r != R[2] and t < 250:
                            act = dst_vi.get_policy(ag)
                            #print(ag, act)
                            env.step(act)
                            r = env.get_reward()
                            ag, g_temp = env.get_observation()
                            reward[k][1] += r
                            t += 1
                            d += get_distance(ag, ag_prev)
                            #env.render()
                            ag_prev = ag
                            
                        distance[k][1] = d
                        time[k][1] = t
                        
                        data.append([reward[k][0], reward[k][1], min_distance[k][0], min_time[k][0], distance[k][0], distance[k][1], time[k][0],  time[k][1], ambiguity[k][0]/5, probability[k][0]/5])
  
                        k += 1
                        with open(path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerows(data)


