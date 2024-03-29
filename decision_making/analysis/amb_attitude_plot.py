import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import csv

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

## functions
def compute_min_time(d):
    return np.ceil(d/(2**(1/2)))


## Params ------------------------------------------------------------------------
alg = [0,1]
dims = [25, 30, 35, 40, 50]
for i in range(len(dims)):
    dims[i] -= 20
# dims = dims - 20
n_trials = 18
D = 50
test_type = 1
ds = 0
p = 0
alpha = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]

if True:
    #fp = "/home/jared/amb_ws/src/ambiguous-decision-making/python/analysis/results/"
    fp = "/home/jared/Data/old_aogs_data/results/"

else:
    fp = None
    
file_name = "ambiguity_attitude_p" + str(p) + "_ds" + str(ds) + ".npy"
path = fp + file_name
file_name = "ambiguity_attitude_p" + str(p) + "_ds" + str(ds+1) + ".npy"
path2 = fp + file_name
# file_name = "alg" + str(0) + "_test" + str(test_type) + "_alpha" + str(1) + "_ds_" + str(0) + ".npy"
# mindpt_path = fp + file_name
# file_name = "alg" + str(1) + "_test" + str(test_type) + "_alpha" + str(0) + "_ds_" + str(0) + ".npy"
# maxd_path = fp + file_name
# file_name = "alg" + str(0) + "_test" + str(test_type) + "_alpha" + str(0) + "_ds_" + str(0) + "final.npy"
# stepsh = fp + file_name
ddim = []
aalpha = []
for i in range(len(dims)):
    aalpha.append(alpha)
for i in range(len(dims)):
    ddim.append(dims[i]*np.ones(len(alpha)))
# print(ddim)
# print(aalpha)
# print(np.shape(ddim))
# print(np.shape(aalpha))

with open(path, 'rb') as f:
    steps_aogs, mind, maxd = np.load(f)
with open(path2, 'rb') as f:
    steps2, mind2, maxd2 = np.load(f)
    
steps_aogs += 1
steps2 += 1
print(steps2)
# print(steps)
temp = steps_aogs.copy()
steps_aogs = np.zeros([25,5,7])
steps_aogs[:,1:5,:] =temp
for i in range(25):
    steps_aogs[i,0,:] = steps2[i]
    
## FP ALG 1 -----------------------------------------------------------------------------
alg = 1
#max_samples = [100, 500, 1e3, 5e3, 1e4]
dims = [25, 30, 35, 40, 50]
n_trials = 25
maxD = 100
test_type = 1
p = 0
timeout = 1
ds = 0
    
alpha = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]

file_name = "ambiguity_attitude_p" + str(p) + "_ds" + str(ds) + "mcgs.npy"
path = fp + file_name

with open(path, 'rb') as f:
    steps_mcgs, mind, maxd = np.load(f)
    
steps_mcgs += 1

# print(steps)
# print("---------------")
# print(mind)
# print("---------------")
# print(maxd)
# print("---------------")
# with open(path, 'rb') as f:
    # data = np.load(f)
a_size = np.shape(steps_aogs)
print(a_size)
steps_aogs_avg = np.zeros([a_size[1],a_size[2]])
# mind_avg = np.zeros([a_size[1],a_size[2]])
# maxd_avg = np.zeros([a_size[1],a_size[2]])
steps_mcgs_avg = np.zeros([a_size[1],a_size[2]])

steps_var = np.zeros([a_size[1],a_size[2]])
# mind_var = np.zeros([a_size[1],a_size[2]])
# maxd_var = np.zeros([a_size[1],a_size[2]])

# print(a_size)      
# print(a_size[1])
# print(a_size[2])

for i in range(a_size[1]):
    for j in range(a_size[2]):
        # print(steps[:,i,j])
        # print(np.shape(steps[:,i]))
        steps_aogs_avg[i][j] = np.average(np.array(steps_aogs[:,i,j]))
        # mind_avg[i][j] = np.average(np.array(mind[:,i,j]))
        # maxd_avg[i][j] = np.average(np.array(maxd[:,i,j]))
        steps_mcgs_avg[i][j] = np.average(np.array(steps_mcgs[:,i,j]))

        steps_var[i][j] = np.var(np.array(steps_aogs[:,i,j]))
        # mind_var[i][j] = np.var(np.array(mind[:,i,j]))
        # maxd_var[i][j] = np.var(np.array(maxd[:,i,j]))
      
# print(mind)  
# print(mind_avg)
print(steps_aogs)
# print(mind_avg)
# print(maxd_avg)
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

fsize = 19
font = {'size'   : fsize}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(1,1,sharey=True,figsize=(6, 7.5))#, gridspec_kw={'width_ratios': [2.45, 3]})


# print(np.min(np.min(d_diff)),np.max(np.max(d_diff)))
ax.contourf(ddim,aalpha, steps_aogs_avg, levels= 20, cmap='Blues', interpolation='hanning', vmin=0, vmax=100)#, cmap='binary')
# plt.xticks(p)
# plt.yticks(amb)
ax.set_ylabel(r"$\alpha$")
ax.set_xlabel("distance")
# ax.title.set_text("AOGS")
# plt.axis('scaled')
ax.tick_params(axis='both', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)
# ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
plt.savefig(fp + "new_figs/aags_trap.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(fp + "new_figs/aags_trap.png", format="png", bbox_inches="tight", pad_inches=0.05)
    
fsize = 18
font = {'size'   : fsize}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(1,1,sharey=True,figsize=(7.5, 7.5))#, gridspec_kw={'width_ratios': [2.45, 3]})


im2 = ax.contourf(ddim,aalpha, steps_mcgs_avg, levels= 20,cmap='Blues', interpolation='hanning', vmin=0, vmax=100)#, cmap='binary')
# plt.xticks(p)
# plt.yticks(amb)
# ax.set_ylabel(r"$\alpha$")
ax.set_xlabel("distance")
# ax.title.set_text("GBOP")
# plt.axis('scaled')
ax.set_yticks([])
fig.colorbar(im2)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)

plt.savefig(fp + "new_figs/gbop_trap.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(fp + "new_figs/gbop_trap.png", format="png", bbox_inches="tight", pad_inches=0.05)

# ax.plot(max_samples, steps_avg)
# ax.plot(max_samples, mind_avg)
# ax.plot(max_samples, maxd_avg)

# ax.fill_between(max_samples, (steps_avg-steps_var), (steps_avg+steps_var), color='b', alpha=.1)
# ax.fill_between(max_samples, (mind_avg-mind_var), (mind_avg+mind_var), color='y', alpha=.1)
# ax.fill_between(max_samples, (maxd_avg-maxd_var), (maxd_avg+maxd_var), color='g', alpha=.1)

# plt.ylabel("Average Reward")
# plt.xlabel("Budget (n)")
# plt.title(title_name)
# plt.legend(["steps", "mind", "maxd"])
# # ax.axis('scaled')
# # plt.autoscale(enable=False)
# # cb = plt.colorbar()
# # plt.legend()


# cb.remove()


# fig, ax = plt.subplots()
# # print(np.min(np.min(d_diff)),np.max(np.max(d_diff)))
# fig = plt.contourf(ddim,aalpha, mind_avg)#, vmin=np.min(np.min(d_diff)), vmax=np.max(np.max(d_diff)))#, cmap='binary')
# # plt.xticks(p)
# # plt.yticks(amb)
# plt.ylabel("alpha")
# plt.xlabel("distance")
# plt.title("MinD")
# # plt.axis('scaled')
# cb = plt.colorbar()

# plt.savefig(fp + "figs/mind.eps", format="eps", bbox_inches="tight", pad_inches=0)
# plt.savefig(fp + "figs/mind.png", format="png", bbox_inches="tight", pad_inches=0.05)
# cb.remove()

# fig = plt.contourf(ddim,aalpha, maxd_avg)#, vmin=np.min(np.min(d_diff)), vmax=np.max(np.max(d_diff)))#, cmap='binary')
# # plt.xticks(p)
# # plt.yticks(amb)
# plt.ylabel("alpha")
# plt.xlabel("distance")
# plt.title("MaxD")
# # plt.axis('scaled')
# # plt.colorbar()
# cb = plt.colorbar()

# plt.savefig(fp + "figs/maxd.eps", format="eps", bbox_inches="tight", pad_inches=0)
# plt.savefig(fp + "figs/maxd.png", format="png", bbox_inches="tight", pad_inches=0.05)
# # plt.legend()
# # fig.colorbar(ax=ax[0], extend='max')
# # plt.show()

# plt.savefig(prefix + "figs/d_diff.eps", format="eps", bbox_inches="tight", pad_inches=0)
# plt.savefig(prefix + "figs/d_diff.png", format="png", bbox_inches="tight", pad_inches=0.05)

# # plt.pause(10)
            

# print(r)

# with open(path, 'rb') as f:
#     np.load(f)