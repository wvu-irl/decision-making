from re import I
import numpy as np
import random
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

def lower_bound(_s, _a, _solver, _delta):
    
    # Solve l for all actions
    l = []
    for a in _s.a_:
        v = 0
        for _ in range(1000):
            kl = KL_divergence_Bernoulli(a.r_,v)
            if(kl <= beta_r(_s.n,_solver.n)/_s.n_ and kl > v):
                v = kl
        l.append(v)

    # solve Bellman Operator for state
    B = np.inf
    for i,a in enumerate(_s.a_):
        P = 0
        C = [] #Solve C!
        for q in C:
            p = 0
            for s in a.s_prime_i_:
                p += q*_solver.graph_[s].L_
            if p > P:
                P = p
            b = l[i] + P
        if b < B:
            B = b

            

def upper_bound(_s, _a, _solver, _delta):
    #min r _ gamma*min_q q*U
    pass

def beta_r(_n, _delta, _B, _K, _H):
    return np.log((3(_B*_K)**_H)/_delta)+np.log(np.e*(1+_n))

def beta_p(_n,_delta,_B,_K,_H):
    return np.log((3(_B*_K)**_H)/_delta)+(_B-1)*np.log(np.e*(1+_n/(_B-1)))

def KL_divergence(_p, _q):
    d = 0
    for p,q in zip(_p,_q):
        d += p*np.log(p/q)

def KL_divergence_Bernoulli(_u,_v):
    return _u*np.log(_u/_v) + (1-_u)*np.log((1-_u)/(1-_v))

def MaxKL(_V,_p,_c): 
    i_p = range(len(_p))
    q = np.zeros(len(_p))
    q_hat = np.zeros(len(_p))
    Z_zero = np.where(_p == 0, i_p)
    Z_hat = np.where(_p > 0, i_p)
    I_star = Z_zero.intersection(np.amax(_V))
    belowBounds = False
    for i in I_star:
        if f(_V[i]) < _c:
            belowBounds = True
            v = _V[i]
            return
    if belowBounds:
        r = 1- np.exp(f(v)-_c)
        for i in I_star:
            q[i] = r/len(I_star) # COME BACK    
    else:
        r = 0
        for i in Z_zero:
            q[i] = 0
        v = newtons_method() # COME BACK
    for i in Z_zero:
        q_hat[i] = _p[i]/(v-_V[i])
    
    q_hat_zero_sum = np.sum(q_hat)
    for i in Z_hat:

        q[i] = ((1-r)*(_p[i]/(v-_V[i])))/q_hat_zero_sum

def f(_v,_V,_p,_Z_hat):
    a = 0
    b = 0
    for i in _Z_hat:
        a += _p[i]*np.log(_v-_V[i])
        b += _p[i]/(_v-_V[i]) 
    return a+np.log(b)
def newtons_method():
    return



    
    

    