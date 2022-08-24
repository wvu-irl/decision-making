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
def count_2_dist(_a, _g, _solver, _is_upper):
    dist = []

    t = 0 # Number of samples
    for n,r,spi in zip(_a.n_, _a.r_, _a.s_prime_i_):
        t += n
        if _is_upper:
            dist.append((n, r+_g*_solver.graph_[spi].U_))
        else:
            dist.append((n, r+_g*_solver.graph_[spi].L_))
    for i in range(len(dist)):
        dist[i] = (dist[i][0]/t, dist[i][1])
    return dist, t

def get_avg(_dist):
    avg = 0
    for el in _dist:
        avg += el[0]*el[1]
    return avg  

def get_confidence(epsilon,t):
    if t == 0:
        return 0
    else:
        # c = t*math.log( 1/((1-epsilon)*2/3) + 1/2)**2/8
        # c = 3/2*(1./(1+np.exp(-c)))-1/2
        c = epsilon*t*math.log( 1/((1-epsilon)*2/3) + 1/2)
        c = 3/2*(1./(1+np.exp(-c)))-1/2
        # c = (-math.log( 1/((1-epsilon)*4/5) + (1/3-1/7)))**2
        # c = (c*t)/(8)
        # c = 3/2*(1./(1+2*np.exp(-c)))-1/2
        if c > 1:
            return 1
        if c < 0:
            return 0
        # print(c)
        return c

## Belief functions --------------------------------
## -------------------------------------------------
## -------------------------------------------------
# Belief functions are a list of pairs(set(rewards),number of counts)
def dist_2_bf(_dist, _t, _epsilon, _l, _u, no_c = False):
    bf, n, epsilon = compute_bf_accuracy(_dist, _epsilon)
    if no_c:
        c = 1
    else:
        c = get_confidence(epsilon, _t)
    return compute_discount_bf(bf, c, _l, _u)

def compute_bf_accuracy(_dist, _e):
    if len(_dist) == 0:
        return _dist, 0, _e
    elif len(_dist) == 1:
        return _dist, 1, _e
    else:
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
            bf[i] = (bf[i][0]-_e, {bf[i][1]})
        if n > 1:
            m = n*_e/comb(n,2)
            for i in range(len(els)):
                for j in range(i):
                    if not(i == j):
                        bf.append((m,{els[i],els[j]}))
        return bf, n, _e

def compute_discount_bf(_bf, _c, _l, _u):
    bf = _bf.copy()
    theta = {_l, _u}
    sum_p = 0
    for i in range(len(bf)):
        sum_p += bf[i][0]*_c
        bf[i] = (bf[i][0]*_c, bf[i][1])
        if type(bf[i][1]) == set:
            temp = bf[i][1]
        else:
            temp = {bf[i][1]}
        for r in  temp:
            if r not in theta:
                theta.add(r)
          
    bf.append((1-_c, theta))
    sum_p += 1-_c
    if sum_p > 1:
        for i in range(len(bf)):   
            bf[i] = (bf[i][0]/sum_p, bf[i][1])
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
    # sum_p = 0
    for el in _bf:
        if type(el[1]) == set:
            E += max(el[1])*el[0]
            # sum_p += el[0]
        else:
            # sum_p += el[0]
            E += el[1]*el[0]
    # print("sum p", sum_p)
    return E