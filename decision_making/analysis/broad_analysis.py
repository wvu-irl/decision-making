import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import json
import pickle

import re
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
test_file = sys.argv[2]
dep_variable = sys.argv[3]
indep_variable = sys.argv[4]

f = open(current + "/../config/analysis/" + var_file +  ".json")
var_config = json.load(f)
f = open(current + "/../config/analysis/" + test_file +  ".json")
test_env = json.load(f)

if test_env["aogs_file"] != "":
    with open(current + "/../analysis/multithread/" + test_env["aogs_file"] , "rb") as f:
            aogs_pickle = pickle.load(f)
if test_env["gbop_file"] != "":
    with open(current + "/../analysis/multithread/" + test_env["gbop_file"] , "rb") as f:
            gbop_pickle = pickle.load(f)
if test_env["uct_file"] != "":
    with open(current + "/../analysis/multithread/" + test_env["uct_file"] , "rb") as f:
            uct_pickle = pickle.load(f)

aogs_data = []
gbop_data = []
uct_data = []
for d in aogs_pickle:
    params = re.split(r'_+', d)
    params.pop()
    
    if len(aogs_pickle[d]["R"]) != 0:
        temp = {}
        temp["epsilon"] = params[0]
        temp["delta"] = params[1]
        temp["horizon"] = params[2]
        temp["samples"] = params[3]
        temp["alpha"] = params[4]
        temp["p"] = params[5]
        temp["init"] = aogs_pickle[d]["init"]
        temp["goal"] = aogs_pickle[d]["goal"]
        temp["r"] = aogs_pickle[d]["R"]
        temp["t"] = aogs_pickle[d]["ts"]
        l = []
        for s1, s2 in zip(aogs_pickle[d]["init"], aogs_pickle[d]["goal"]):
            l.append(get_distance(s1,s2))
        temp["d"] = l
        
        aogs_data.append(temp)
    
for d in gbop_pickle:
    params = re.split(r'_+', d)
    params.pop()
    
    if len(gbop_pickle[d]["R"]) != 0:
        temp = {}
        temp["epsilon"] = params[0]
        temp["delta"] = params[1]
        temp["horizon"] = params[2]
        temp["samples"] = params[3]
        temp["alpha"] = params[4]
        temp["p"] = params[5]
        temp["init"] = gbop_pickle[d]["init"]
        temp["goal"] = gbop_pickle[d]["goal"]
        temp["r"] = gbop_pickle[d]["R"]
        temp["t"] = gbop_pickle[d]["ts"]
        l = []
        for s1, s2 in zip(gbop_pickle[d]["init"], gbop_pickle[d]["goal"]):
            l.append(get_distance(s1,s2))
        temp["d"] = l        
        gbop_data.append(temp)
    

for d in uct_pickle:
    params = re.split(r'_+', d)
    params.pop()
    
    if len(uct_pickle[d]["R"]) != 0:
        temp = {}
        temp["horizon"] = params[0]
        temp["samples"] = params[1]
        temp["c"] = params[2]
        temp["p"] = params[3]
        temp["init"] = uct_pickle[d]["init"]
        temp["goal"] = uct_pickle[d]["goal"]
        temp["r"] = uct_pickle[d]["R"]
        temp["t"] = uct_pickle[d]["ts"]
        l = []
        for s1, s2 in zip(uct_pickle[d]["init"], uct_pickle[d]["goal"]):
            l.append(get_distance(s1,s2))
        temp["d"] = l        
        uct_data.append(temp)
            

x = var_config[indep_variable]

data = {}
for el in x:
    data[str(el)] = []
    
aogs_y = copy.deepcopy(data)
gbop_y = copy.deepcopy(data)
uct_y = copy.deepcopy(data)

for el in aogs_data:
    aogs_y[el[indep_variable]] += el[dep_variable]
# for el in gbop_data:
#     gbop_y[el[indep_variable]].append(el[dep_variable])
# for el in uct_data:
#     uct_y[el[indep_variable]].append(el[dep_variable])

y = []
for el in x:
    print(aogs_y[str(el)])
    y.append(np.average(np.array(aogs_y[str(el)])))

 
plt.plot(x,y) 
plt.show()
plt.pause(1)
while 1:
    plt.pause(1)  
# plt.ylabel("Average Reward")
# plt.xlabel("Budget (n)")
# plt.title(title_name)
# plt.legend(["aogs_c", "aogs_o", "ucb", "gbop"])
# ax.axis('scaled')
# plt.autoscale(enable=False)
# cb = plt.colorbar()
# plt.legend()

# plt.savefig(fp + "figs/" + title_name + "_avg_reward.eps", format="eps", bbox_inches="tight", pad_inches=0)
# plt.savefig(fp + "figs/" + title_name + "_avg_reward.png", format="png", bbox_inches="tight", pad_inches=0.05)
# cb.remove()

# cross_ref -> alg, param, p
# indep_variable -> param, d, horizon, samples, p, e,d
# dep_variable -> r, t


#will have separate file to compare and collect data on grid tunnel  



# aogs_c_avg = np.zeros(len(aogs_opt_data[0]))
# aogs_o_avg = np.zeros(len(aogs_opt_data[0]))
# ucb_avg = np.zeros(len(ucb_data[0]))
# gbop_avg = np.zeros(len(gbop_data[0]))

# aogs_c_var = np.zeros(len(aogs_opt_data[0]))
# aogs_o_var = np.zeros(len(aogs_opt_data[0]))
# ucb_var = np.zeros(len(ucb_data[0]))
# gbop_var = np.zeros(len(gbop_data[0]))

            
# for i in range(len(aogs_opt_data[0])):
#     aogs_c_avg[i] = np.average(np.array(aogs_con_data[i]))
#     aogs_o_avg[i] = np.average(np.array(aogs_opt_data[i]))
#     ucb_avg[i] = np.average(np.array(ucb_data[i]))
#     gbop_avg[i] = np.average(np.array(gbop_data[i]))

#     aogs_c_var[i] = np.var(np.array(aogs_con_data[i]))
#     aogs_o_var[i] = np.var(np.array(aogs_opt_data[i]))
#     ucb_var[i] = np.var(np.array(ucb_data[i]))
#     gbop_var[i] = np.var(np.array(gbop_data[i]))

# ## PRINT --------------------------------------------------------

# #fig, ax = plt.subplots(1,2,sharey='row',figsize=(7.5, 3.75 ))
# # fig, axs = plt.subplots(1)
# # fig = plt.contourf(amb,p, t_diff, vmin=np.min(np.min(t_diff)), vmax=np.max(np.max(t_diff)))
# # ax[0].set_xticks(p)
# # ax[0].set_yticks(amb)

# if test_type == 0:
#     title_name = "Grid World"
# elif test_type == 1:
#     title_name = "Grid Trap"
# else:
#     title_name = "Sailing" 

# fig, ax = plt.subplots()

# ax.plot(max_samples, aogs_c_avg)
# ax.plot(max_samples, aogs_o_avg)
# ax.plot(max_samples, ucb_avg)
# ax.plot(max_samples, gbop_avg)


# # ax.fill_between(max_samples, (aogs_c_avg-aogs_c_var), (aogs_c_avg+aogs_c_var), color='b', alpha=.1)
# # ax.fill_between(max_samples, (aogs_o_avg-aogs_o_var), (aogs_o_avg+aogs_o_var), color='y', alpha=.1)
# # ax.fill_between(max_samples, (ucb_avg-ucb_var), (ucb_avg+ucb_var), color='g', alpha=.1)
# # ax.fill_between(max_samples, (gbop_avg-gbop_var), (gbop_avg+gbop_var), color='g', alpha=.1)


# plt.ylabel("Average Reward")
# plt.xlabel("Budget (n)")
# plt.title(title_name)
# plt.legend(["aogs_c", "aogs_o", "ucb", "gbop"])
# # ax.axis('scaled')
# # plt.autoscale(enable=False)
# # cb = plt.colorbar()
# # plt.legend()

# plt.savefig(fp + "figs/" + title_name + "_avg_reward.eps", format="eps", bbox_inches="tight", pad_inches=0)
# plt.savefig(fp + "figs/" + title_name + "_avg_reward.png", format="png", bbox_inches="tight", pad_inches=0.05)
# # cb.remove()

# # # print(np.min(np.min(d_diff)),np.max(np.max(d_diff)))
# # fig = plt.contourf(amb,p, d_diff, vmin=np.min(np.min(d_diff)), vmax=np.max(np.max(d_diff)))#, cmap='binary')
# # # plt.xticks(p)
# # # plt.yticks(amb)
# # plt.ylabel("Transition probability p")
# # plt.xlabel("Transition ambiguity c")
# # plt.title("Percent increase in distance")
# # # plt.axis('scaled')
# # plt.colorbar()

# # # plt.legend()
# # # fig.colorbar(ax=ax[0], extend='max')
# # # plt.show()

# # plt.savefig(prefix + "figs/d_diff.eps", format="eps", bbox_inches="tight", pad_inches=0)
# # plt.savefig(prefix + "figs/d_diff.png", format="png", bbox_inches="tight", pad_inches=0.05)

# # # plt.pause(10)