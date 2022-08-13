from cmath import inf
import math
from math import comb
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#prefix = "/media/jared/32GB/home/ambiguity_ws/src/ambiguous-decision-making/python/analysis/data/"
prefix = "/home/jared/ambiguity_ws/src/ambiguous-decision-making/python/analysis/data/"
## UTILITIES ---------------------------------------
## -------------------------------------------------
## -------------------------------------------------
def update_model(_dist, _obs):
    dist = _dist.copy()
    i = 0
    while i < len(dist):
        if dist[i][1] == _obs:
            temp = (dist[i][0]+1,_obs)
            dist[i] = temp
            return dist
        i += 1
    dist.append((1,_obs))
    return dist

def count_2_dist(_dist):
    dist = _dist.copy()

    n = 0
    for el in dist:
        n += el[0]
    for i in range(len(dist)):
        temp = (dist[i][0]/n, dist[i][1])
        dist[i] = temp
    return dist, n

def get_avg(_dist):
    avg = 0
    for el in _dist:
        avg += el[0]*el[1]
    return avg

def get_ind(_dist, _n):
    p = 0
    i = 0
    while p <= _n:
        p += _dist[i][0]
        i += 1
    r = _dist[i-1][1]
    return i, r    

## Belief functions --------------------------------
## -------------------------------------------------
## -------------------------------------------------
# Belief functions are a list of pairs(set(rewards),number of counts)
def compute_bf_accuracy(_dist, _e):
    bf = _dist.copy()

    els = []
    for el in bf:
        #print(el[1])
        els.append(el[1])
        if el[0]-_e < 0 or el[0]+_e > 1:
            _e = np.min([el[0], 1-el[0]])
    els = np.unique(els)
    n = len(els)
    
    for i in range(len(bf)):
        bf[i] = (bf[i][0]-_e, bf[i][1])
    if n > 1:
        m = n*_e/comb(n,2)
        for i in range(len(els)):
            for j in range(len(els)):
                if not(i == j):
                    bf.append((m,{els[i],els[j]}))

    return bf, n, _e

def compute_discount_bf(_bf, _c, _l, _u):
    bf = _bf.copy()
    theta = {_l, _u}
    for i in range(len(bf)):
        bf[i] = (bf[i][0]*_c, bf[i][1])
        if type(bf[i][1]) == set:
            temp = bf[i][1]
        else:
            temp = {bf[i][1]}
        for r in  temp:
            if r not in theta:
                theta.add(r)
                
    bf.append((1-_c, theta))    
    return bf

def lower_expectation(_bf):
    E = 0
    for el in _bf:
        if type(el[1]) == set:
            E += min(el[1])*el[0]
        else:
            E += el[1]*el[0]
    return E

def upper_expectation(_bf):
    E = 0
    for el in _bf:
        if type(el[1]) == set:
            E += max(el[1])*el[0]
        else:
            E += el[1]*el[0]
    return E

## Get action ---------------------------------------
## --------------------------------------------------
## --------------------------------------------------

def get_action_epsilon_greedy(_actions, _e, _rng):
    n = _rng.random()
    if n < _e:
        return _rng.choice(len(_actions))
    else:
        Q_s = -inf
        ind = 0
        for i in range(len(_actions)):
            dist, N = count_2_dist(_actions[i])
            Q = get_avg(dist)
            if Q > Q_s or Q == 0:
                Q_s = Q
                ind = i
        return ind
        

        
def get_action_ucb1(_actions, _c):
    ucb_max = -inf
    ind = 0
    N = 0
    for i in range(len(_actions)):
        for j in range(len(_actions[i])):
            N += _actions[i][j][0]
            
    for i in range(len(_actions)):
        dist, n = count_2_dist(_actions[i])
        if n == 0:
            return i
        Q = get_avg(dist)
        ucb = Q + _c*np.sqrt(np.log(N)/n)
        if ucb > ucb_max:
            ucb_max = ucb
            ind = i
    return ind

def get_action_amb_e(_actions, _e, _alpha, _l, _u):
    exp_max = -inf
    ind = 0
    # N = 0
    # for i in range(len(_actions)):
    #     for j in range(len(_actions[i])):
    #         N += _actions[i][j][0]
            
    for i in range(len(_actions)):
        dist, t = count_2_dist(_actions[i])
        # print("|||||||||||||||||||||||||")
        # print(t)
        # print(dist)
        bf, n, e = compute_bf_accuracy(dist, _e)
        # print(bf)
        c = (-math.log( 1/((1-e)*4/5) + (1/3-1/7)))**2
        if n == 0:
            c = 5/4*(1./(1+2*n*np.exp(-np.inf*np.sign(c)))-(1/3-(1/7)))
        else:
            c = (c*t)/(8*n)
            c = 5/4*(1./(1+2*n*np.exp(-c))-(1/3-(1/7)))
        
        bf = compute_discount_bf(bf, c, _l, _u)
        # print(bf)
        # print("--------------")
        low_exp = lower_expectation(bf)
        up_exp = upper_expectation(bf)
        expectation = _alpha*low_exp + (1-_alpha)*up_exp #+ 0.5**np.sqrt(np.log(N)/t)
        if expectation > exp_max:
            exp_max = expectation
            ind = i
    return ind


def get_action_amb_c(_actions, _c, _alpha, _l, _u):
    pass

def get_action_amb_entropy(_actions, _alpha, _l, _u):
    pass

## Entropy Measures----------------------------------
## --------------------------------------------------
## --------------------------------------------------

## -------------------------------------------------
## -------------------------------------------------
## -------------------------------------------------
## -------------------------------------------------
## Initialize params
# Assume we are using epsilon-greedy
num_el = 10
num_trials = 2000
num_iter = 200
num_a = 10
num_outcomes = 10
L = -3
U = 3
rng = np.random.default_rng()


## Values for each approach
# e-greedy 
epsilon = [0]* num_el
for i in range(len(epsilon)): epsilon[i] = i/num_el

# ucb1
# c = [0]* num_el
# for i in range(len(c)): c[i] = 1/2**((num_el/2 - i))
c = [ 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2]

# ambiguity
alpha = [0]* num_el
for i in range(len(alpha)): alpha[i] = i/num_el
# epsilon = [0.1]
# c = [1]
# alpha = [0.9]

# ## Models
# temp =[]
# for el in R:
#     temp.append((0,el))
# dist = [temp.copy] * num_a

# True model 
#t_model = dist
# dim-1, action, dim-2 outcomes


## Data
e_greedy_r = np.zeros([int(num_trials),int(num_iter), num_el])
ucb_r = np.zeros([int(num_trials),int(num_iter), num_el])
amb_r = np.zeros([int(num_trials),int(num_iter), num_el])
amb_e_r = np.zeros([int(num_trials),int(num_iter), num_el])


e_greedy_opt = np.zeros([int(num_trials),int(num_iter), num_el])
ucb_opt = np.zeros([int(num_trials),int(num_iter), num_el])
amb_opt = np.zeros([int(num_trials),int(num_iter), num_el])
amb_e_opt = np.zeros([int(num_trials),int(num_iter), num_el])


##  Sample and select
for i in list(range(int(num_trials))):
    print ("Trial ", i)
    opt_a = 0
    opt_r = -50
    # Resample true model 
    t_model = [None]*num_a
    for j in range(len(t_model)):
        n = rng.integers(1,num_outcomes)
        r = rng.uniform(L,U,n)
        p = rng.uniform(0,1,n)
        p /= sum(p)
        temp = [None] * len(p)
        for k in range(len(p)):
            temp[k] = (p[k], r[k])
        t_model[j] = temp
        # print(temp)
        avg = get_avg(temp)
        # print(avg)
        if avg > opt_r:
            opt_a = j
            opt_r = avg
    # print(opt_a)
    
    # e-greedy model
    e_models = [None] * num_el
    # ucb1 model
    ucb_models = [None] * num_el
    # ambiguity model
    amb_models = [None] * num_el
    
    amb_e_models = [None] * num_el

    
    temp = [None] * num_a
    for j in range(num_a):
        temp[j] = []
    
    for j in range(num_el):
        e_models[j] = temp
        ucb_models[j] = temp
        amb_models[j] = temp
        amb_e_models[j] = temp   
    
    ## Learn
    for j in list(range(int(num_iter))):
        #print("Trial ", i, " Iteration ", j)
        # sample all actions
        outcomes = [0] * num_a
        for k in range(num_a):
            p = rng.uniform()
            ind, outcomes[k] = get_ind(t_model[k], p)

        # update models
        # e-greedy
        for k in range(num_el):
            act = get_action_epsilon_greedy(e_models[k], epsilon[k], rng)
            if act == opt_a:
                e_greedy_opt[i][j][k] += 1
            e_models[k][act] = update_model(e_models[k][act], outcomes[act])
            e_greedy_r[i][j][k] = outcomes[act]
        # ucb1
        for k in range(num_el):
            act = get_action_ucb1(ucb_models[k], c[k])
            if act == opt_a:
                ucb_opt[i][j][k] += 1
            ucb_models[k][act] = update_model(ucb_models[k][act], outcomes[act])
            ucb_r[i][j][k] = outcomes[act]
        # amb
        for k in range(num_el):
            act = get_action_amb_e(amb_models[k], 0.005, alpha[k], L, U)
            if act == opt_a:
                amb_opt[i][j][k] += 1
            amb_models[k][act] = update_model(amb_models[k][act], outcomes[act])
            amb_r[i][j][k] = outcomes[act]
            
        # amb_fixed alpha            
        for k in range(num_el):
            act = get_action_amb_e(amb_models[k], epsilon[k], 0.75, L, U)
            if act == opt_a:
                amb_e_opt[i][j][k] += 1
            amb_e_models[k][act] = update_model(amb_e_models[k][act], outcomes[act])
            amb_e_r[i][j][k] = outcomes[act]


## Need to also compare across epsilon values, maximizing entropy

## Plot

e_greedy_avg = np.average(e_greedy_r,0)
ucb_avg = np.average(ucb_r,0)
amb_avg = np.average(amb_r,0)
amb_e_avg = np.average(amb_e_r,0)

# print(amb_avg[0])
# print(amb_avg[len(amb_avg[:][0])])
# avg rewards

iter = list(range(int(num_iter)))
# print(np.shape(e_greedy_avg))
# print(np.shape(iter))
# print(np.shape(epsilon))


fig = plt.contourf(epsilon, iter, e_greedy_avg)
plt.xlabel("Epsilon")
plt.ylabel("Iteration")
plt.title("Avg Reward, epsilon")
# plt.axis('scaled')
plt.colorbar()
# plt.show()
plt.savefig(prefix + "eps_avg_r.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "eps_avg_r.png", format="png", bbox_inches="tight", pad_inches=0.05)
plt.clf()

fig = plt.plot(iter, e_greedy_avg[:,0])
plt.xlabel("t")
plt.ylabel("r")
plt.title("Avg Reward, epsilon")
fig = plt.plot(iter, e_greedy_avg[:,1])
fig = plt.plot(iter, e_greedy_avg[:,9])
plt.legend(["0", "0.1", "0.9"])
# plt.axis('scaled')
# plt.colorbar()
# plt.show()
plt.savefig(prefix + "eps.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "eps.png", format="png", bbox_inches="tight", pad_inches=0.05)
plt.clf()


fig = plt.contourf(c, iter, ucb_avg)
plt.xlabel("c")
plt.ylabel("Iteration")
plt.title("Avg Reward, ucb")
# plt.axis('scaled')
plt.colorbar()
# plt.show()
plt.savefig(prefix + "ucb_avg_r.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "ucb_avg_r.png", format="png", bbox_inches="tight", pad_inches=0.05)
plt.clf()


fig = plt.contourf(alpha, iter, amb_avg)
plt.xlabel("alpha")
plt.ylabel("Iteration")
plt.title("Avg Reward, ambiguity")
# plt.axis('scaled')
plt.colorbar()
# plt.show()
plt.savefig(prefix + "amb_avg_r.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "amb_avg_r.png", format="png", bbox_inches="tight", pad_inches=0.05)
plt.clf()

fig = plt.contourf(epsilon, iter, amb_e_avg)
plt.xlabel("epsilon")
plt.ylabel("Iteration")
plt.title("Avg Reward, ambiguity e")
# plt.axis('scaled')
plt.colorbar()
# plt.show()
plt.savefig(prefix + "amb_e_avg_r.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "amb_e_avg_r.png", format="png", bbox_inches="tight", pad_inches=0.05)
plt.clf()
# cb.remove()
##
##
##
##

e_greedy_avg_opt = np.average(e_greedy_opt,0)
ucb_avg_opt = np.average(ucb_opt,0)
amb_avg_opt = np.average(amb_opt,0)
amb_e_avg_opt = np.average(amb_e_opt,0)

print(amb_avg_opt[0])
print(amb_avg_opt[len(amb_avg_opt[:][0])])
# avg rewards

iter = list(range(int(num_iter)))
print(np.shape(e_greedy_avg_opt))
print(np.shape(iter))
print(np.shape(epsilon))


fig = plt.contourf(epsilon, iter, e_greedy_avg_opt)
plt.xlabel("Epsilon")
plt.ylabel("Iteration")
plt.title("Avg Opt Action, epsilon")
# plt.axis('scaled')
plt.colorbar()
# plt.show()
plt.savefig(prefix + "eps_avg_o.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "eps_avg_o.png", format="png", bbox_inches="tight", pad_inches=0.05)
plt.clf()

fig = plt.contourf(c, iter, ucb_avg_opt)
plt.xlabel("c")
plt.ylabel("Iteration")
plt.title("Avg Opt Action, ucb")
# plt.axis('scaled')
plt.colorbar()
# plt.show()
plt.savefig(prefix + "ucb_avg_o.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "ucb_avg_o.png", format="png", bbox_inches="tight", pad_inches=0.05)
plt.clf()

fig = plt.contourf(alpha, iter, amb_avg_opt)
plt.xlabel("alpha")
plt.ylabel("Iteration")
plt.title("Avg Opt Action, ambiguity")
# plt.axis('scaled')
plt.colorbar()
# plt.show()
plt.savefig(prefix + "amb_avg_o.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "amb_avg_o.png", format="png", bbox_inches="tight", pad_inches=0.05)
plt.clf()

fig = plt.contourf(epsilon, iter, amb_e_avg_opt)
plt.xlabel("epsilon")
plt.ylabel("Iteration")
plt.title("Avg Opt Action, ambiguity e")
# plt.axis('scaled')
plt.colorbar()
# plt.show()
plt.savefig(prefix + "amb_e_avg_o.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "amb_e_avg_o.png", format="png", bbox_inches="tight", pad_inches=0.05)
plt.clf()

fig = plt.contourf(alpha, iter, ucb_avg_opt - amb_avg_opt)
plt.xlabel("alpha/c")
plt.ylabel("Iteration")
plt.title("Diff Opt Action")
# plt.axis('scaled')
plt.colorbar()
# plt.show()
plt.savefig(prefix + "ucb_amb_err.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "ucb_amb_err.png", format="png", bbox_inches="tight", pad_inches=0.05)
plt.clf()

fig = plt.plot(epsilon, e_greedy_avg_opt[num_iter-1])
fig = plt.plot(c, ucb_avg_opt[num_iter-1])
fig = plt.plot(alpha, amb_avg_opt[num_iter-1])
plt.xlabel("e,c,a")
plt.ylabel("% optimal actions")
plt.title("Optimal Actions, All")

plt.legend(["e-greedy", "ucb", "amb"])
# plt.axis('scaled')
# plt.colorbar()
# plt.show()
plt.savefig(prefix + "all_opt.eps", format="eps", bbox_inches="tight", pad_inches=0)
plt.savefig(prefix + "all_opt.png", format="png", bbox_inches="tight", pad_inches=0.05)
plt.clf()