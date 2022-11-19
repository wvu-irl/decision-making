import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import json
import pickle

import numpy as np
import random
import copy
import itertools


import matplotlib.pyplot as plt

## functions
def get_distance(s1, s2):
        return ((s1[0]-s2[0])**2 + (s1[1]-s2[1])**2)**0.5
    
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

# aogs_keys = []
# temp = []
# temp.append(var_config["epsilon"])
# temp.append(var_config["delta"])
# temp.append(var_config["horizon"])
# temp.append(var_config["max_samples"])
# temp.append(var_config["alpha"])
# temp.append(var_config["probabilities"])
# temp = list(itertools.product(*temp))
 
# for el in temp:
#     data_key = ""
#     for itm in el:
#         data_key += str(itm) + "_"
#     aogs_keys.append(data_key)


# temp = copy.deepcopy(aogs_data)
# for el in aogs_data: 
#     add_el = True
#     if el["alg"]["model_accuracy"]["epsilon"] not in var_config["epsilon"]:
#         add_el = False
#     if el["alg"]["model_accuracy"]["delta"] not in var_config["delta"]:
#         add_el = False
#     if el["alg"]["search"]["horizon"] not in var_config["horizon"]:
#         add_el = False
#     if el["alg"]["search"]["max_samples"] not in var_config["max_samples"]:
#         add_el = False
#     if el["alg"]["action_selection"]["params"]["alpha"] not in var_config["alpha"]:
#         add_el = False
#     if el["env"]["params"]["p"] not in var_config["probabilities"]:
#         add_el = False      
        
#     if add_el:
#         temp.append(el)
# aogs_data = copy.deepcopy(aogs_data)

# temp = copy.deepcopy(gbop_data)
# for el in gbop_data: 
#     add_el = True
#     if el["alg"]["model_accuracy"]["epsilon"] not in var_config["epsilon"]:
#         add_el = False
#     if el["alg"]["model_accuracy"]["delta"] not in var_config["delta"]:
#         add_el = False
#     if el["alg"]["search"]["horizon"] not in var_config["horizon"]:
#         add_el = False
#     if el["alg"]["search"]["max_samples"] not in var_config["max_samples"]:
#         add_el = False
#     if el["alg"]["action_selection"]["move_params"]["alpha"] not in var_config["alpha"]:
#         add_el = False
#     if el["env"]["params"]["p"] not in var_config["probabilities"]:
#         add_el = False      
        
#     if add_el:
#         temp.append(el)
# gbop_data = copy.deepcopy(gbop_data)

# temp = copy.deepcopy(uct_data)
# for el in uct_data: 
#     add_el = True
#     if el["alg"]["search"]["horizon"] not in var_config["horizon"]:
#         add_el = False
#     if el["alg"]["search"]["max_samples"] not in var_config["max_samples"]:
#         add_el = False
#     if el["alg"]["action_selection"]["decision_params"]["c"] not in var_config["c"]:
#         add_el = False
#     if el["env"]["params"]["p"] not in var_config["probabilities"]:
#         add_el = False      
        
#     if add_el:
#         temp.append(el)
# uct_data = copy.deepcopy(uct_data)

#for each load corresponding mt_config params (except distance which must be computed)
#bin according to param. 
aogs_keys = []
gbop_keys = []
uct_keys = []

aogs_rewards = {}
aogs_ts = {}
aogs_init = {}
aogs_goal = {}
aogs_dist = {}

gbop_rewards = {}
gbop_ts = {}
gbop_init = {}
gbop_goal = {}
gbop_dist = {}

uct_rewards = {}
uct_ts = {}
uct_init = {}
uct_goal = {}
uct_dist = {}

if indep_variable == "distance":
    pass
else:
    if indep_variable == "param":
        indep_key = "alpha"
        indep_key_uct = "c"
    elif indep_variable == "samples":
        indep_key = "n_samples"
        indep_key_uct = "n_samples"
    elif indep_variable == "horizon":
        indep_key = "horizon"
        indep_key_uct = "horizon"
        
    for el in var_config[indep_key]:
        aogs_rewards[str(el)] = []
        aogs_ts[str(el)] = []
        aogs_init[str(el)] = []
        aogs_goal[str(el)] = []
        aogs_dist[str(el)] = []
    for el in var_config[indep_key]:
        gbop_rewards[str(el)] = []
        gbop_ts[str(el)] = []
        gbop_init[str(el)] = []
        gbop_goal[str(el)] = []
        gbop_dist[str(el)] = []
    for el in var_config[indep_key_uct]:
        uct_rewards[str(el)] = []
        uct_ts[str(el)] = []
        uct_init[str(el)] = []
        uct_goal[str(el)] = []
        uct_dist[str(el)] = []
        
    for el in var_config[indep_key]:
        temp = []
        temp.append(var_config["epsilon"])
        temp.append(var_config["delta"])
        temp.append(var_config["horizon"])
        temp.append(var_config["max_samples"])
        temp.append([el])
        temp.append(var_config["probabilities"])
        temp = list(itertools.product(*temp))
        
        for itm in temp:
            data_key = ""
            for param in itm:
                data_key += str(param) + "_"
            
            aogs_rewards[str(el)].append(aogs_data[data_key]["R"])
            aogs_ts[str(el)].append(aogs_data[data_key]["ts"])
            aogs_init[str(el)].append(aogs_data[data_key]["init"])
            aogs_goal[str(el)].append(aogs_data[data_key]["goal"])

            gbop_rewards[str(el)].append(gbop_data[data_key]["R"])
            gbop_ts[str(el)].append(gbop_data[data_key]["ts"])
            gbop_init[str(el)].append(gbop_data[data_key]["init"])
            gbop_goal[str(el)].append(gbop_data[data_key]["goal"])
        
        aogs_keys.append(temp)
        gbop_keys.append(temp)
        
    for el in var_config[indep_key_uct]:
        temp = []
        temp.append(var_config["horizon"])
        temp.append(var_config["max_samples"])
        temp.append([el])
        temp.append(var_config["probabilities"])
        temp = list(itertools.product(*temp))
        
        for itm in temp:
            data_key = ""
            for param in itm:
                data_key += str(param) + "_"
            
            uct_rewards[str(el)].append(uct_data[data_key]["R"])
            uct_ts[str(el)].append(uct_data[data_key]["ts"])
            uct_init[str(el)].append(uct_data[data_key]["init"])
            uct_goal[str(el)].append(uct_data[data_key]["goal"])
        
        uct_keys.append(temp)
            
# do distance computation

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