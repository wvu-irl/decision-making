from cmath import inf
import math
from math import comb
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations

## TODO 
# finish restrict_dist()

#generate bf should take a confidence, use this to compute accuracy
# Need way to generate the bel() and plausibility() of each element absed on
# this. Use it to compute a "b" vector.
# compute masses and convert to distribution
# then do as before

invA = []
MAX_NUMEL = 12
            
def powerset(max_el):
    l = list(range(max_el))
    pset = []
    for i in range(0,max_el+1):
        for element in combinations(l,i):
            pset.append(element)
    return pset
         
def generate_invA(M):
    for i in range(2,M+1):
        #print("---------")
        numel = (2**(i))-(i+1)
        A = np.ones([i+1,numel])
        #print("i ", i, numel)
        n = 0
        for el in powerset(i):
            #print("els ", el)
            if len(el) > 1:
                for k in range(i):
                    if k not in el:
                        #print("check ", k,el)
                        # print(k, numel-n-1)
                        A[k][n] = 0
                n += 1
        #print(A)
        invA.append(np.linalg.pinv(A))
        #print(np.linalg.pinv(A))
        

generate_invA(MAX_NUMEL)


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
            # print(r+_g*_solver.graph_[spi].U_)
        else:
            dist.append((n, r+_g*_solver.graph_[spi].L_))
            # print(r+_g*_solver.graph_[spi].L_)
    for i in range(len(dist)):
        dist[i] = (dist[i][0]/t, dist[i][1])
    return dist, t

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
    
def get_accuracy(_delta,_t, a):
    change = inf
    epsilon = 0
    while np.fabs(change) > 0.005:
        e = -math.log(1/ (2/3*(1-_delta+1/2) ) - 1)
        e /= (_t*(math.log(1/ (2/3*(1-epsilon) ) + 1/2))**2)
        #e = 1-e
        change = e-epsilon
        # print("------------------")
        # print (epsilon, change, e)
        # print(a*change)
        epsilon = epsilon + a*change
        # print (epsilon, change, e)
    #epsilon += 1    
    if epsilon > 1:
        return 1
    if epsilon < 0:
        return 0   
    return epsilon

## Belief functions --------------------------------
## -------------------------------------------------
## -------------------------------------------------
# Belief functions are a list of pairs(set(rewards),number of counts)
def bin_dist(_dist, _n = MAX_NUMEL-1):
    temp_dist = []
    l_dist = len(_dist)
    mass = np.zeros(l_dist)
    val = np.zeros(l_dist)
    # print(val)
    
    for i in range(l_dist):
        mass[i] = _dist[i][0]
        val[i] = _dist[i][1]
    
    if len(_dist) < _n:
        return mass, val
    else:
        # print("-------------------")
        # print(mass)
        min_val = min(val)
        bin_size = (max(val)-min_val)/(_n-1)
        if bin_size != 0:
            tmass = np.zeros(_n)
            tval = np.zeros(_n)
            for i in range(l_dist):
                ind = int(np.floor((val[i]-min_val)/bin_size))
                # print(val[i]-min_val)
                tval[ind] = tmass[ind]*tval[ind] + mass[i]*val[i]
                tmass[ind] += mass[i] 
                tval[ind] /= tmass[ind]  
            return tmass, tval   
        else:
            
            return mass, val                
        

def generate_bf_conf(_dist, _delta, _t, _l, _u, _e):
    if len(_dist) == 0:
        _dist.append(1, {_l, _u})
        return _dist
    elif len(_dist) == 1:
        c = get_confidence(_e,_t)
        _dist[0] = (c, _dist[0][1])
        _dist.append((1-c, {_l, _u}))
        # print(c, _t)
        return _dist
    else:
        # print(len(_dist))
        # print("-----------")
        epsilon = get_accuracy(_delta,_t, 0.05)
        
        mass, val = bin_dist(_dist)
        
        mass /= np.sum(mass)
        # print(mass)
        # print(val)
        lmass = len(mass)
        bel = np.zeros([lmass,1])
        pl = np.zeros([lmass,1])
        pl_minus_bel = np.zeros([lmass+1,1])
        temp_sum = 0
        for i in range(lmass):
            bel[i] = max([0, mass[i]-epsilon])
            pl[i] = min([1, mass[i]+epsilon])
            pl_minus_bel[i] = pl[i] - bel[i]
            temp_sum += bel[i]
            
        pl_minus_bel[lmass] = 1-temp_sum

        #compute belief + plausibility
        #compute excess mass from belief
        
        #generate
        # print(len(invA))
        # print(invA[len(mass)-2])
        # print(invA[len(mass)-2])
        # print(pl_minus_bel)
        # print(len(mass)-2)
        mass = (1-_delta)*np.matmul(invA[len(mass)-2],pl_minus_bel) 
        # print(" <0 ", mass < 0)
        #print(mass)
        # print("mass", lmass)
        pset = powerset(lmass)
        del pset[0:lmass+1]
        # print(pset)
        theta = {_l, _u}
        lpset = len(pset)
        # print(pset)
        for i in range(len(val)):
            theta.add(val[i])
        
        sum_p = 0
        bf = []
        #singletons
        for i in range(len(bel)):
            # print(val, val[i], type(val[i]))
            # print(val[i], val[i][0])
            # print(type(val[i]))
            bf.append((bel[i], val[i]))
            sum_p += bel[i]
        # multiple hypostheses
        for i in range(lpset):
            temp = set()
            for j in pset[i]:
                # print(pset[i])
                # print(val[j])
                temp.add(val[j])
            bf.append((mass[i], temp))
            sum_p += mass[i]
        #compute theta 
        bf.append((_delta, theta))
        sum_p += _delta
        # p > 1 check
        if sum_p > 1:
            for i in range(len(bf)):   
                bf[i] = (float(bf[i][0]/sum_p), bf[i][1])
            # print("------------")
            # print(bf)
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
    # print("------------")
    # print(_bf)
    for el in _bf:
        if type(el[1]) == set:
            # print(max(el[1])*el[0])
            E += max(el[1])*el[0]
            # print(max(el[1]))
            # sum_p += el[0]
        else:
            # print(el[1]*el[0])
            # sum_p += el[0]
            E += el[1]*el[0]

    # print(E)    
    return E