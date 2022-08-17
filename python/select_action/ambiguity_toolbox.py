from cmath import inf
import math
from math import comb
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

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
