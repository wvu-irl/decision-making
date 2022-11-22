import sys
import os

from numpy import var
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
if indep_variable == "alpha":
    uct_indep = "c"
else:
    uct_indep = indep_variable
if len(sys.argv) > 5:
    crossref_variable = sys.argv[5]
else:
    crossref_variable = None

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
        temp["max_samples"] = params[3]
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
        temp["max_samples"] = params[3]
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
        temp["max_samples"] = params[1]
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

# for el in aogs_data: 
#     print(el)
var_config["d"] = []

for x in aogs_data:
    for d in x["d"]:
        # print(d)
        if d not in var_config["d"]:
            var_config["d"].append(d)
for x in gbop_data:
    for d in x["d"]:
        if d not in var_config["d"]:
            var_config["d"].append(d)
for x in uct_data:
    for d in x["d"]:
        if d not in var_config["d"]:
            var_config["d"].append(d)
var_config["d"].sort()

# var_config["d"] = list(range(min(var_config["d"]), max(var_config["d"]), (max(var_config["d"])-min(var_config["d"]))/100 ))
            
### Plot
x = var_config[indep_variable]
if indep_variable == "alpha":
    x_uct = var_config[uct_indep]
else:
    x_uct = x.copy()

if crossref_variable == None:
    num_plots = 1
else:
    num_plots = len(var_config[crossref_variable]) + 1

splt_len = [int(np.ceil(np.sqrt(num_plots))), int(np.floor(np.sqrt(num_plots)))]
fig, ax = plt.subplots(splt_len[0],splt_len[1],figsize=(7.5, 7.5 ))
# fig.title(dep_variable)
# print(ax)
print(splt_len)
print(num_plots)
if crossref_variable != None:
    trials = copy.deepcopy(var_config[crossref_variable])
    trials.append(copy.deepcopy(var_config[crossref_variable]))

for i in range(num_plots):
    
    data = {}
    for el in x:
        data[str(el)] = []
        
    aogs_temp = copy.deepcopy(data)
    gbop_temp = copy.deepcopy(data)
    uct_temp = {}
    data = {}
    for el in x_uct:
        uct_temp[str(el)] = []
    
    if crossref_variable != None:
        if type(trials[i]) == list:
            filtered_vars = trials[i]
        else:
            filtered_vars = [trials[i]]

        for el in aogs_data:
            # print(el)
            # print(type(el[crossref_variable]), type(filtered_vars[0]),(el[crossref_variable] in filtered_vars))
            if float(el[crossref_variable]) in filtered_vars:
                # print(dep_variable)
                # print(el[dep_variable])
                # for itm in el[indep_variable]:
                #     if itm in aogs_temp:
                aogs_temp[el[indep_variable]] += el[dep_variable]
        for el in gbop_data:
            if float(el[crossref_variable]) in filtered_vars:
                # for itm in el[indep_variable]:
                #     if itm in gbop_temp:
                gbop_temp[el[indep_variable]] += el[dep_variable]
        for el in uct_data:
            if indep_variable not in ["epsilon", "delta"] and float(el[crossref_variable]) in filtered_vars:
                # for itm in el[uct_indep]:
                #     if itm in uct_temp:
                uct_temp[el[uct_indep]] += el[dep_variable]
    else:
        for el in aogs_data:
            # for itm in el[indep_variable]:
            #     if itm in aogs_temp:
            aogs_temp[el[indep_variable]] += el[dep_variable]
        for el in gbop_data:
            # for itm in el[indep_variable]:
            #     if itm in gbop_temp:
            gbop_temp[el[indep_variable]] += el[dep_variable]
        for el in uct_data:
            if indep_variable not in ["epsilon", "delta"]:
                    # for itm in el[indep_variable]:
                    #     if itm in uct_temp:
                    uct_temp[el[uct_indep]] += el[dep_variable]
                
    # print("------------")
    # print(crossref_variable)
    # for el in aogs_temp:
    #     print(el)

    aogs_y = []
    gbop_y = []
    uct_y = []

    for el in x:
        # print("----", aogs_temp)
        aogs_y.append(np.average(np.array(aogs_temp[str(el)])))
        print("aogs", len(aogs_temp[str(el)]))
        gbop_y.append(np.average(np.array(gbop_temp[str(el)])))
        print("gbop", len(gbop_temp[str(el)]))
    if indep_variable not in ["epsilon", "delta"]:
        for el in x_uct:
            print("uct", len(uct_temp[str(el)]))
            uct_y.append(np.average(np.array(uct_temp[str(el)])))
    # print(x)
    # print(aogs_y)
    if num_plots == 1:
        ax.plot(x,aogs_y)
        ax.plot(x,gbop_y)
        if indep_variable not in ["epsilon", "delta"]: 
            ax.plot(x_uct,uct_y) 
        
        ax.set_title(str(indep_variable + " vs " + dep_variable))
        ax.legend(["aogs", "gbop", "uct"])
    else:
        # print([int(np.floor(i/splt_len[0])), int(i % splt_len[0])])
        x_ind = int(i % splt_len[0])
        y_ind = int(np.floor(i/splt_len[0]))
        
        ax[x_ind, y_ind].plot(x,aogs_y)
        ax[x_ind, y_ind].plot(x,gbop_y) 
        if indep_variable not in ["epsilon", "delta"]: 
            ax[x_ind, y_ind].plot(x_uct,uct_y) 
        
        ax[x_ind, y_ind].set_title(crossref_variable + " " + str(trials[i]))
        ax[x_ind, y_ind].legend(["aogs", "gbop", "uct"])
  
    
    
plt.show()
while 1:
    plt.pause(1)

fig.savefig(current + "/plots/" + indep_variable + "_vs_" + dep_variable + "cr_" + crossref_variable + ".eps", format="eps", bbox_inches="tight", pad_inches=0)
fig.savefig(current + "/plots/" + indep_variable + "_vs_" + dep_variable + "cr_" + crossref_variable + ".png", format="png", bbox_inches="tight", pad_inches=0.0)