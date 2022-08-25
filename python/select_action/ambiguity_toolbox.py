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
        

#generate_invA(MAX_NUMEL = 12)


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
def bin_dist(_dist, _n = MAX_NUMEL):
    if len(_dist) > _n:
        pass
    else:
        temp_dist = []
        l_dist = len(_dist)
        mass = np.zeros([l_dist,1])
        val = np.zeros([l_dist,1])
        for i in range(l_dist):
            mass[i] = _dist[i][0]
            val[i] = _dist[i][1]
            
        min_mass = min(mass)
        bin_size = max(mass)-min_mass
        
        tmass = np.zeros([l_dist,1])
        tval = np.zeros([l_dist,1])
        for i in range(l_dist):
           ind = np.floor((tmass-min_mass)/bin_size)
           tval[ind] = tmass[ind]*tval[ind] + mass[i]*val[i]
           tmass[ind] += mass[i] 
           tval[ind] /= tmass[ind]     
           pass         
        # for i in range(l_dist):
        #     temp_dist.append(tmass[i],tval[i])          
        return tmass, tval

def generate_bf_conf(_dist, _delta, _t, _l, _u):
    if len(_dist) == 0:
        _dist.append(1, {_l, _u})
        return _dist
    elif len(_dist) == 1:
        _dist[0] = (1-_delta, _dist[1])
        _dist.append(_delta, {_l, _u})
        return _dist
    else:
        epsilon = get_accuracy(_delta,_t, 0.05)
        
        mass, val = bin_dist(_dist)
        
        mass /= np.sum(mass)
        
        lmass = len(mass)
        bel = np.zeros([lmass,1])
        pl = np.zeros([lmass,1])
        pl_minus_bel = np.zeros([lmass+1,1])
        temp_sum = 0
        for i in range(lmass):
            bel[i] = max([0, mass[i]-epsilon])
            pl[i] = max([1, mass[i]+epsilon])
            pl_minus_bel[i] = pl[i] - bel[i]
            temp_sum += pl[i] - bel[i]
            
        pl_minus_bel[lmass+1] = temp_sum

        #compute belief + plausibility
        #compute excess mass from belief
        
        #generate
        
        mass = (1-_delta)*np.matmul(invA[len(mass)-2],pl_minus_bel) 

        pset = powerset(len(mass))
        pset.remove(0)
        
        theta = {_l, _u}
        lpset = len(pset)
        for el in pset(lpset):
            theta.add(el)
        
        sum_p = 0
        bf = []
        #singletons
        for i in range(bel):
            bf.append((bel[i], val[i]))
            sum_p += bel[i]
        # multiple hypostheses
        for i in range(lpset):
            bf.append((mass[i], pset[i]))
            sum_p += mass[i]
        #compute theta 
        bf.append((_delta, theta))
        sum_p += _delta
        # p > 1 check
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