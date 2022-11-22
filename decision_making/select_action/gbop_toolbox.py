from cmath import nan
from re import I
import re
from unicodedata import name
import numpy as np
import random
import sys
import os
import matplotlib.pyplot as plt
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
def boundSolver(_s , _solver, _type):
    # Solve l for all actions
    bound = -np.inf
    totalActions = _solver.search_params_["branch_factor_s"]#len(_s.a_)
    actions = _s.a_
    random.shuffle(actions)
    # print("parent", _s.s_, hash(str(_s.s_)))
    for a in actions:
        if len(a.s_prime_i_) > 0:
            #Solve for U for State-Action
            Cr = beta_r(a.N_,1-_solver.alg_params_["model_accuracy"]["delta"],_solver.search_params_["branch_factor_s"],totalActions,_solver.search_params_["horizon"])/a.N_
            reward = np.sum(a.r_)/len(a.r_)
            if _type == "upper": 
                kl = lambda x, A : KL_divergence_Bernoulli(reward,x) - Cr
                dkl = lambda x, A : dKL_divergence_Bernoulli(reward,x)
            else:
                kl = lambda x, A : -KL_divergence_Bernoulli(reward,x) + Cr
                dkl = lambda x, A : -dKL_divergence_Bernoulli(reward,x)
                
            b_ul = [reward,1] if _type == "upper" else [0,reward]
            ul = newtons_method(kl,dkl,bounds=b_ul)
            Cp = beta_p(a.N_,1-_solver.alg_params_["model_accuracy"]["delta"],_solver.search_params_["branch_factor_s"],totalActions,_solver.search_params_["horizon"])/a.N_
            _, T, _ = a.get_transition_model()

            p = 0
            B = []
            if _type == "upper":
                for s in a.s_prime_:
                    # print("child", s, hash(str(s)))
                    B.append(_solver.graph_[_solver.gi_[hash(str(s))]].U_)
            else:
                for s in a.s_prime_:
                    B.append(_solver.graph_[_solver.gi_[hash(str(s))]].L_)
            C = MaxKL(B,T,Cp,_solver.bounds_)
            for j,q in enumerate(C):
                p += q*B[j]
            b = ul + _solver.alg_params_["gamma"]*p
            if b > bound:
                bound = b
    if bound == -np.inf:
        if _type == "upper":
            bound = _s.U_
        else:
            bound = _s.L_
    return bound

def beta_r(_n, _delta, _B, _K, _H):
    return np.log((3*(_B*_K)**_H)/_delta) + np.log(np.e*(1+_n))

def beta_p(_n,_delta,_B,_K,_H):
    return  np.log((3*(_B*_K)**_H)/_delta) + (_B-1)*np.log(np.e*(1+_n/(_B-1)))

def KL_divergence(_p, _q):
    d = 0
    for p,q in zip(_p,_q):
        d += p*np.log(p/q)

def KL_divergence_Bernoulli(_u,_v):
    a = 0
    b =  np.infty
    if _v > 0:
        if (_u/_v) > 0:
            a = _u*np.log(_u/_v)
    
    if 1-_v > 0:
        if ((1-_u)/(1-_v)) > 0:
            b = (1-_u)*np.log((1-_u)/(1-_v))
    return a + b

def dKL_divergence_Bernoulli(_u,_v):
    if _v != 0 or _v != 1: 
        return  (1-_u)/(1-_v) - _u/_v
    else:
        return np.infty

def MaxKL(_V,_p,_c,_bounds): 
    if all(np.isclose(_p, 0)):
        _p = np.ones(_p.size)/_p.size
    A = {"V": _V,"p":_p,"c":_c,"Z_hat": None}
    def f(_v,_args):
        a = 0
        b = 0
        for i in _args["Z_hat"]:
            if  _v-_args["V"][i] > 0:
                a += _args["p"][i]*np.log(_v-_args["V"][i])
                b += _args["p"][i]/(_v-_args["V"][i])
            else:
                return -np.infty
        return a+np.log(b)-_args["c"]

    def f_prime(_v,_args):
        a = 0
        b = 0
        for i in _args["Z_hat"]:
            a += _args["p"][i]/(_v-_args["V"][i])
            b += _args["p"][i]/(_v-_args["V"][i])**2
            return a - (b/a)

    # Solving for Z
    q = np.zeros(_p.shape[0])
    q_hat = np.zeros(len(_p))
    Z_zero, = np.where(_p == 0)
    Z_hat, = np.where(_p > 0)
    A["Z_hat"] = Z_hat.tolist()
    I_star = []
    for i in Z_zero:
        if i == np.argmax(_V):
            I_star.append(i)

    # Solving for r, v and some q* values
    belowBounds = False
    for i in I_star:
        if f(_V[i],A) < 0:
            belowBounds = True
            v = _V[i]
            break    
    if belowBounds:
        r = 1 - np.exp(f(v,_V,_p,Z_hat))
        for i in I_star:
            q[i] = r/len(I_star)
        for i in Z_zero:
            if i not in I_star:
                q[i] = 0
    else:
        r = 0
        for i in Z_zero:
            q[i] = 0
        sigma = np.sum(_p*np.square(_V)) - np.square(np.sum(_p*_V))
        v = newtons_method(f,f_prime,A,[np.amax(_V), _bounds[1]],initVal = np.amax(_V) +1) 
    #Solving for q*
    for i in Z_hat:
        q_hat[i] = _p[i]/(v-_V[i])
    
    q_hat_zero_sum = np.sum(q_hat)
    for i in Z_hat:
        q[i] = ((1-r)*(_p[i]/(v-_V[i])))/q_hat_zero_sum
    return q


def newtons_method(func,dfunc,_args = None,bounds = [0, 1],accuracy =.1,N = 1000,divExept = .001,weight = .9, initVal = None):
    if initVal == None:
        xNext = (bounds[1]-bounds[0])/2
    else:
        xNext = initVal

    xList = [xNext]
    dxList = []
    for _ in range(N):
        x = xNext
        # print(x)
        fx = func(x,_args)
        dfx = dfunc(x,_args)
        if np.isnan(dfx) or dfx == -np.inf:
            dfx = dfunc(x+divExept,_args)
        xNext = x-fx/dfx

        if xNext < bounds[0]:
            xNext = weight*bounds[0] + (1-weight)*x
        elif xNext > bounds[1]:
            xNext = weight*bounds[1] +(1-weight)*x
        
        xList.append(xNext)
        dxList.append(dfx)
        if np.abs(x-xNext) < accuracy:
            return xNext
    return xNext
