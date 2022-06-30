import numpy as np
import sys
import csv
print(csv.__file__)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.rcParams['text.usetex'] = True
# Initialize --------------------------
# User specified -------
world_size = 10
num = 7
n_trials = 25
p = [0.2, 0.4, 0.6, 0.8, 1]
amb = [0, 0.2, 0.4, 0.6, 0.8, 1]
# Auto populated -------
prefix = "/home/jared/pomdp_ws/src/ambiguity-value-iteration/data/"

raw_data = []

#store 
t_vi = []
t_avi = []
t_min = []
d_vi = []
d_avi = []
d_min = []
r_vi = []
r_avi = []

# Separate data -----------------------
for i in range(num):
    file_name = "/w" + str(world_size) + "avi" + str(i+1) + ".csv"
    path = prefix + file_name
    with open(path, "r", newline="") as f:
        
        _csv = csv.reader(f, delimiter=',')
        for row in _csv:
            if row[0] != 'r_vi':
                raw_data.append(row)
                
for i in range(len(p)):
    t_vi_row = []
    t_avi_row = []
    t_min_row = []
    d_vi_row = []
    d_avi_row = []
    d_min_row = []
    r_vi_row = []
    r_avi_row = []
    for j in range(len(amb)):
        t_vi_data = []
        t_avi_data = []
        t_min_data = []
        d_vi_data = []
        d_avi_data = []
        d_min_data = []
        r_vi_data = []
        r_avi_data = []
        for row in raw_data:
            #print(row[9], p[i],row[8],amb[j], float(row[9]) == p[i] and float(row[8]) == amb[j])
            if float(row[9]) == p[i] and float(row[8]) == amb[j]:
                r_vi_data.append(float(row[0]))
                r_avi_data.append(float(row[1]))
                d_min_data.append(float(row[2]))
                t_min_data.append(float(row[3]))
                d_vi_data.append(float(row[4]))
                d_avi_data.append(float(row[5]))
                t_vi_data.append(float(row[6]))
                t_avi_data.append(float(row[7]))
        t_vi_row.append(t_vi_data)
        t_avi_row.append(t_avi_data)
        t_min_row.append(t_min_data)
        d_vi_row.append(d_vi_data)
        d_avi_row.append(d_avi_data)
        d_min_row.append(d_min_data)
        r_vi_row.append(r_vi_data)
        r_avi_row.append(r_avi_data)
    t_vi.append(t_vi_row)
    t_avi.append(t_avi_row)
    t_min.append(t_min_row)
    d_vi.append(d_vi_row)
    d_avi.append(d_avi_row)
    d_min.append(d_min_row)
    r_vi.append(r_vi_row)
    r_avi.append(r_avi_row)
                

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