import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import csv

import numpy as np
import random

import gym

from gym_envs.gridworld import GridWorld
from gym_envs.gridtrap import GridTrap
from gym_envs.sailing import Sailing
from solvers.aogs import AOGS
from solvers.uct import UCT
# from solvers.mcgs import MCGS
from select_action import actions as act

## functions
def compute_min_time(d):
    return np.ceil(d/(2**(1/2)))


## Params ------------------------------------------------------------------------
alg = 0
max_samples = [5e3]#[100, 500, 1e3, 5e3, 1e4]
n_trials = 18
D = 50
test_type = 0
ds = 0
    
alpha = 1

if True:
    fp = "/home/jared/ambiguity_ws/src/ambiguous-decision-making/python/analysis/results/"
else:
    fp = None
    
file_name = "alg" + str(alg) + "_test" + str(test_type) + "_alpha" + str(alpha) + "_ds_" + str(ds) + "last18_5k.npy"
path = fp + file_name
data = []


with open(path, 'rb') as f:
    data = np.load(f)

t_vi_avg = np.zeros([len(p),len(amb)])
t_avi_avg = np.zeros([len(p),len(amb)])
t_min_avg = np.zeros([len(p),len(amb)])
d_vi_avg = np.zeros([len(p),len(amb)])
d_avi_avg = np.zeros([len(p),len(amb)])
d_min_avg = np.zeros([len(p),len(amb)])
r_vi_avg = np.zeros([len(p),len(amb)])
r_avi_avg = np.zeros([len(p),len(amb)])

t_vi_var = np.zeros([len(p),len(amb)])
t_avi_var = np.zeros([len(p),len(amb)])
t_min_var = np.zeros([len(p),len(amb)])
d_vi_var = np.zeros([len(p),len(amb)])
d_avi_var = np.zeros([len(p),len(amb)])
d_min_var = np.zeros([len(p),len(amb)])
r_vi_var = np.zeros([len(p),len(amb)])
r_avi_var = np.zeros([len(p),len(amb)])
            
for i in range(len(p)):
    for j in range(len(amb)):
        t_vi_avg[i][j] = np.average(np.array(t_vi[i][j]))
        t_avi_avg[i][j] = np.average(np.array(t_avi[i][j]))
        t_min_avg[i][j] = np.average(np.array(t_min[i][j]))
        d_vi_avg[i][j] = np.average(np.array(d_vi[i][j]))
        d_avi_avg[i][j] = np.average(np.array(d_avi[i][j]))
        d_min_avg[i][j] = np.average(np.array(d_min[i][j]))
        r_vi_avg[i][j] = np.average(np.array(r_vi[i][j]))
        r_avi_avg[i][j] = np.average(np.array(r_avi[i][j]))

        t_vi_var[i][j] = np.var(np.array(t_vi[i][j]))
        t_avi_var[i][j] = np.var(np.array(t_avi[i][j]))
        t_min_var[i][j] = np.var(np.array(t_min[i][j]))
        d_vi_var[i][j] = np.var(np.array(d_vi[i][j]))
        d_avi_var[i][j] = np.var(np.array(d_avi[i][j]))
        d_min_var[i][j] = np.var(np.array(d_min[i][j]))
        r_vi_var[i][j] = np.var(np.array(r_vi[i][j]))
        r_avi_var[i][j] = np.var(np.array(r_avi[i][j]))

# print(t_avi)
# Plot --------------------------------

t_diff = (t_avi_avg - t_vi_avg)/ t_vi_avg* 100
# print(t_diff)
d_diff = (d_avi_avg - d_vi_avg)/ d_vi_avg* 100
# print(d_diff)

#fig, ax = plt.subplots(1,2,sharey='row',figsize=(7.5, 3.75 ))
# fig, axs = plt.subplots(1)
fig = plt.contourf(amb,p, t_diff, vmin=np.min(np.min(t_diff)), vmax=np.max(np.max(t_diff)))
# ax[0].set_xticks(p)
# ax[0].set_yticks(amb)
plt.ylabel("Transition probability p")
plt.xlabel("Transition ambiguity c")
plt.title("Percent increase in time")
# plt.axis('scaled')
# plt.autoscale(enable=False)
cb = plt.colorbar()
# plt.legend()

plt.savefig(prefix + "figs/t_diff.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "figs/t_diff.png", format="png", bbox_inches="tight", pad_inches=0.05)
cb.remove()

# print(np.min(np.min(d_diff)),np.max(np.max(d_diff)))
fig = plt.contourf(amb,p, d_diff, vmin=np.min(np.min(d_diff)), vmax=np.max(np.max(d_diff)))#, cmap='binary')
# plt.xticks(p)
# plt.yticks(amb)
plt.ylabel("Transition probability p")
plt.xlabel("Transition ambiguity c")
plt.title("Percent increase in distance")
# plt.axis('scaled')
plt.colorbar()

# plt.legend()
# fig.colorbar(ax=ax[0], extend='max')
# plt.show()

plt.savefig(prefix + "figs/d_diff.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "figs/d_diff.png", format="png", bbox_inches="tight", pad_inches=0.05)

# plt.pause(10)