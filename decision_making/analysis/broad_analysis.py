import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import json
import pickle

import numpy as np
import random

import matplotlib.pyplot as plt

## functions
def compute_min_time(d):
    return np.ceil(d/(2**(1/2)))

var_file = sys.argv[1]
test_env = sys.argv[2]
dep_variable = sys.argv[3]
indep_variable = sys.argv[4]

f = open(current + "/../config/multithread/" + var_file +  ".json")
var_config = json.load(f)

if test_env["aogs_file"] != "":
    with open(current + "/analysis/multithread/" + test_env["aogs_file"] , "rb") as f:
            aogs_data = pickle.load(f)
if test_env["gbop_file"] != "":
    with open(current + "/analysis/multithread/" + test_env["gbop_file"] , "rb") as f:
            gbop_data = pickle.load(f)
if test_env["uct_file"] != "":
    with open(current + "/analysis/multithread/" + test_env["uct_file"] , "rb") as f:
            uct_data = pickle.load(f)

# for each dataset remove elements not contained in mt_config

#for each load corresponding mt_config params (except distance which must be computed)
#bin according to param. 
if indep_variable == "param":
    pass
elif indep_variable == "samples":
    pass
elif indep_variable == "horizon":
    pass    
elif indep_variable == "distance":
    pass

#do reward and num timesteps to reach goal
if dep_variable == "ed":
    pass
elif dep_variable == "alg":
    #compare each alg with best performance
    pass
elif dep_variable == "raw":
    # compare reward of each alg with their own params
    pass  

#will have separate file to compare and collect data on grid tunnel  


aogs_c_avg = np.zeros(len(aogs_opt_data[0]))
aogs_o_avg = np.zeros(len(aogs_opt_data[0]))
ucb_avg = np.zeros(len(ucb_data[0]))
gbop_avg = np.zeros(len(gbop_data[0]))

aogs_c_var = np.zeros(len(aogs_opt_data[0]))
aogs_o_var = np.zeros(len(aogs_opt_data[0]))
ucb_var = np.zeros(len(ucb_data[0]))
gbop_var = np.zeros(len(gbop_data[0]))

            
for i in range(len(aogs_opt_data[0])):
    aogs_c_avg[i] = np.average(np.array(aogs_con_data[i]))
    aogs_o_avg[i] = np.average(np.array(aogs_opt_data[i]))
    ucb_avg[i] = np.average(np.array(ucb_data[i]))
    gbop_avg[i] = np.average(np.array(gbop_data[i]))

    aogs_c_var[i] = np.var(np.array(aogs_con_data[i]))
    aogs_o_var[i] = np.var(np.array(aogs_opt_data[i]))
    ucb_var[i] = np.var(np.array(ucb_data[i]))
    gbop_var[i] = np.var(np.array(gbop_data[i]))

## PRINT --------------------------------------------------------

#fig, ax = plt.subplots(1,2,sharey='row',figsize=(7.5, 3.75 ))
# fig, axs = plt.subplots(1)
# fig = plt.contourf(amb,p, t_diff, vmin=np.min(np.min(t_diff)), vmax=np.max(np.max(t_diff)))
# ax[0].set_xticks(p)
# ax[0].set_yticks(amb)

if test_type == 0:
    title_name = "Grid World"
elif test_type == 1:
    title_name = "Grid Trap"
else:
    title_name = "Sailing" 

fig, ax = plt.subplots()

ax.plot(max_samples, aogs_c_avg)
ax.plot(max_samples, aogs_o_avg)
ax.plot(max_samples, ucb_avg)
ax.plot(max_samples, gbop_avg)


# ax.fill_between(max_samples, (aogs_c_avg-aogs_c_var), (aogs_c_avg+aogs_c_var), color='b', alpha=.1)
# ax.fill_between(max_samples, (aogs_o_avg-aogs_o_var), (aogs_o_avg+aogs_o_var), color='y', alpha=.1)
# ax.fill_between(max_samples, (ucb_avg-ucb_var), (ucb_avg+ucb_var), color='g', alpha=.1)
# ax.fill_between(max_samples, (gbop_avg-gbop_var), (gbop_avg+gbop_var), color='g', alpha=.1)


plt.ylabel("Average Reward")
plt.xlabel("Budget (n)")
plt.title(title_name)
plt.legend(["aogs_c", "aogs_o", "ucb", "gbop"])
# ax.axis('scaled')
# plt.autoscale(enable=False)
# cb = plt.colorbar()
# plt.legend()

plt.savefig(fp + "figs/" + title_name + "_avg_reward.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(fp + "figs/" + title_name + "_avg_reward.png", format="png", bbox_inches="tight", pad_inches=0.05)
# cb.remove()

# # print(np.min(np.min(d_diff)),np.max(np.max(d_diff)))
# fig = plt.contourf(amb,p, d_diff, vmin=np.min(np.min(d_diff)), vmax=np.max(np.max(d_diff)))#, cmap='binary')
# # plt.xticks(p)
# # plt.yticks(amb)
# plt.ylabel("Transition probability p")
# plt.xlabel("Transition ambiguity c")
# plt.title("Percent increase in distance")
# # plt.axis('scaled')
# plt.colorbar()

# # plt.legend()
# # fig.colorbar(ax=ax[0], extend='max')
# # plt.show()

# plt.savefig(prefix + "figs/d_diff.eps", format="eps", bbox_inches="tight", pad_inches=0)
# plt.savefig(prefix + "figs/d_diff.png", format="png", bbox_inches="tight", pad_inches=0.05)

# # plt.pause(10)