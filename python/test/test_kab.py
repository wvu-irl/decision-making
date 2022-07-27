
from cmath import inf
import math
from math import comb
import numpy as np
import random


## UTILITIES ---------------------------------------
## -------------------------------------------------
## -------------------------------------------------
def count_2_dist(_dist):
    n = 0
    for el in _dist:
        n += n[1]
    for i in range(len(_dist)):
        _dist[1] /= n    
    return _dist, n

def get_avg(_dist):
    avg = 0
    for el in _dist:
        avg += el[0]*el[1]
    return avg

def get_ind(_dist, _n):
    p = 0
    i = 0
    while p < _n:
        p += _dist[i][0]
        i += 1
    return i   

## Belief functions --------------------------------
## -------------------------------------------------
## -------------------------------------------------
# Belief functions are a list of pairs(set(rewards),number of counts)
def compute_bf_accuracy(_dist, _e):
    els = []
    for el in _dist:
        els.append(el[0])
        if el[1]-_e < 0 or el[1]+_e > 1:
            _e = np.min(el[1], 1-el[1])
    els = np.unique(els)
    n = len(els)
    
    for i in range(len(_dist)):
        _dist[i][1] -= _e
    
    m = n*_e/comb(n,2)
    for i in range(len(els)):
        for j in range(len(els)):
            if not(i == j):
                _dist.append({els[i],els[j]}, m)

    return _dist, n, _e

def compute_discount_bf(_bf, _c, _l, _u):
    theta = {_l, _u}
    for i in range(len(_bf)):
        _bf[i][1] *= _c
        for r in  _bf[i][0]:
            if r not in theta:
                theta.append(r)
                
    _bf.append((theta,1-_c))    
    return _bf

def lower_expectation(_bf):
    E = 0
    for el in _bf:
        E += np.min(el[0])*el[1]
    return E

def upper_expectation(_bf):
    E = 0
    for el in _bf:
        E += np.max(el[0])*el[1]
    return E

## Get action ---------------------------------------
## --------------------------------------------------
## --------------------------------------------------

def get_action_epsilon_greedy(_actions, _e, _rng):
    n = _rng.random()
    if n < _e:
        Q_s = -inf
        ind = 0
        for i in range(len(_actions)):
            dist, N = count_2_dist(_actions[i])
            Q = get_avg(dist)
            if Q > Q_s:
                Q_s = Q
                ind = i
        return ind
    else:
        return _rng.choice(0,len(_actions))

        
def get_action_ucb1(_actions, _c):
    ucb_max = -inf
    ind = 0
    N = 0
    for i in range(len(_actions)):
        for j in range(len(_actions[i])):
            N += _actions[i][0]
            
    for i in range(len(_actions)):
        dist, n = count_2_dist(_actions[i])
        Q, n = get_avg(dist)
        ucb = Q + _c*np.sqrt(np.log(N)/n)
        if ucb > ucb_max:
            ucb_max = ucb
            ind = i
    return ind

def get_action_amb_e(_actions, _e, _alpha, _l, _u):
    exp_max = -inf
    ind = 0
    for i in range(len(_actions)):
        dist, t = count_2_dist(_actions[i])
        bf, n, e = compute_bf_accuracy(dist, _e)
        
        c = (-math.log( 1/((1-e)*4/5) + (1/3-1/7)))**2;
        c = (c*t)/(8*n);
        c = 5/4*(1./(1+2*n*exp(-temp))-(1/3-(1/7)));
        
        bf = compute_discount_bf(bf, c, _l, _u)
        low_exp = lower_expectation(bf)
        up_exp = upper_expectation(bf)
        exp = _alpha*low_exp + (1-_alpha)*up_exp
        if exp > exp_max:
            exp_max = exp
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
num_trials = 1e3
num_iter = 1e3
num_a = 5
num_outcomes = 10
R = list(range(0,5,25))
rng = np.random.default_rng()


## Values for each approach
# e-greedy 
epsilon = [0]* num_el
for i in range(len(epsilon)): epsilon[i] = i/num_el

# ucb1
c = [0]* num_el
for i in range(len(c)): c[i] = i**2*0.5

# ambiguity
alpha = [0]* num_el
for i in range(len(alpha)): alpha[i] = i/num_el


## Models
temp =[]
for el in R:
    temp.append((0,el))
dist = [temp.copy] * num_a

# True model 
t_model = dist
# dim-1, action, dim-2 outcomes

# e-greedy model
e_models = [dist] * num_el
# ucb1 model
ucb_models = [dist] * num_el
# ambiguity model
amb_models = [dist] * num_el


## Data
e_greedy_r = np.zeros(num_trials,num_iter, num_el)
ucb_r = np.zeros(num_trials,num_iter, num_el)
amb_r = np.zeros(num_trials,num_iter, num_el)

# e_greedy_r = np.zeros(num_trials,num_iter, num_el)
# ucb_r = np.zeros(num_trials,num_iter, num_el)
# amb_r = np.zeros(num_trials,num_iter, num_el)



##  Sample and select
for i in list(range(num_trials)):
    # Resample true model 
    for j in range(len(t_model)):
        temp = rng.choice(R, 1000)
        side, count = np.unique(temp,return_counts=True)
        p = count / len(temp)
        for k in range(len(R)):
            t_model[j][k][0] = p[k]
            
    ## Learn
    for j in list(range(num_iter)):
        print ("Trial ", i, " | Iteration ", j)
        # sample all actions

        # update models


## Plot

# avg rewards
