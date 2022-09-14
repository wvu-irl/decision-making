import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import csv

import numpy as np
import random

import gym

from envs.gridworld import GridWorld
from envs.gridtrap import GridTrap
from envs.sailing import Sailing
from planners.aogs import AOGS
from planners.uct import UCT
# from solvers.mcgs import MCGS
from select_action import actions as act

## functions
def compute_min_time(d):
    return np.ceil(d/(2**(1/2)))


## Params ------------------------------------------------------------------------
alg = 0
#n_trials = 200
#D = 50
test_type = 0
ds = 0  
alpha = 1

if True:
    fp = "/home/jared/ambiguity_ws/src/ambiguous-decision-making/python/analysis/results/"
else:
    fp = None
    
file_name = "alg" + str(alg) + "_test" + str(test_type) + "_alpha" + str(alpha) + "_ds_" + str(ds) + ".npy"
path1 = fp + file_name
file_name = "alg" + str(alg) + "_test" + str(test_type) + "_alpha" + str(alpha) + "_ds_" + str(ds) + "1k_last7.npy"
path2 = fp + file_name
file_name = "alg" + str(alg) + "_test" + str(test_type) + "_alpha" + str(alpha) + "_ds_" + str(ds) + "_final.npy"
path_f = fp + file_name


data1 = []
data2 = []

# OPEN -------------------------------------
with open(path1, 'rb') as f:
    data1 = np.load(f)
    
with open(path2, 'rb') as f:
    data2 = np.load(f)
    
# MERGE ------------------------------------
print("before-------------")
print(data1)
data1[3][50:55] = data2[2][1:5]
print("after|||||||||||||")
print(data1)

# SAVE -------------------------------------

with open(path_f, 'wb') as f:
    np.save(f, data1)
            