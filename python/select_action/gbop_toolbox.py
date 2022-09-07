from re import I
import numpy as np
import random
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# def lower_bound(_s, _a,_solver, _delta):
    
#     # Solve l for all actions
#     l = []
#     for a in _s.a_:
#         v = 0
#         Cr = beta_r(_s.N_,_solver.n_,_solver.B_,len(_s.a_),_solver.H_)
#         r = np.sum(a.r_)/len(a.r_)
#         kl = lambda x,A : KL_divergence_Bernoulli(r,x) - Cr
#         dkl = lambda x,A : dKL_divergence_Bernoulli(r,x)
#         v = newtons_method(kl,dkl)
#         l.append(v) #COME BACK

#     # solve Bellman Operator for state
#     B = np.inf
#     for i,a in enumerate(_s.a_):
#         P = 0
#         Cp = beta_p(_s.N_,_solver.n_,_solver.B_,len(_s.a_),_solver.H_) 
#         s_prime, T,reward = _s.a_.get_transition_model()
#         C = MaxKL([_solver.graph_[_solver.gi_[s]].V_ for s in s_prime],T,Cp)
#         for q in C:
#             p = 0
#             for s in a.s_prime_i_:
#                 p += q*_solver.graph_[_solver.gi_[s]].L_
#             if p > P:
#                 P = p
#             b = l[i] + P
#         if b < B:
#             B = b
    
#     # Find Low Bound
#     lowerBound = np.inf
#     for low in l:
#        L = low + _solver.gamma_*B
#        if  L < lowerBound:
#             lowerBound = L
#     _s.L_ = lowerBound
#     return lowerBound

            

def boundSolver(_s , _solver, _type):
    # Solve l for all actions
    bound = np.inf
    totalActions = len(_s.a_)
    for a in _s.a_:
        #Solve for U for State-Action
        Cr = beta_r(_s.N_,_solver.n_,_solver.B_,totalActions,_solver.H_)
        reward = np.sum(a.r_)/len(a.r_)
        kl = lambda x,A : KL_divergence_Bernoulli(reward,x) - Cr
        dkl = lambda x,A : dKL_divergence_Bernoulli(reward,x)
        u = newtons_method(kl,dkl)

        #Solve for Sum(p*U) for state action
        Cp = beta_p(_s.N_,_solver.n_,_solver.B_,totalActions,_solver.H_) 
        V, T, reward = a.get_transition_model()
        C = MaxKL(V[0],T,Cp)

        p = 0
        if _type == "upper":
            for j,q in enumerate(C):
                p += q*_solver.graph_[_solver.gi_[a.s_prime_i_[j]]].U_
            b = u + _solver.gamma_*p
            if b > bound:
                bound = b
        else:
            for j,q in enumerate(C):
                p += q*_solver.graph_[_solver.gi_[a.s_prime_i_[j]]].L_
            b = u + _solver.gamma_*p
            if b > bound:
                bound = b
    return bound

def beta_r(_n, _delta, _B, _K, _H):
    return np.log((3*(_B*_K)**_H)/_delta)+np.log(np.e*(1+_n))

def beta_p(_n,_delta,_B,_K,_H):
    return np.log((3*(_B*_K)**_H)/_delta)+(_B-1)*np.log(np.e*(1+_n/(_B-1)))

def KL_divergence(_p, _q):
    d = 0
    for p,q in zip(_p,_q):
        d += p*np.log(p/q)

def KL_divergence_Bernoulli(_u,_v):
    return _u*np.log(_u/_v) + (1-_u)*np.log((1-_u)/(1-_v))

def dKL_divergence_Bernoulli(_u,_v):
    return (1-_u)/(1-_v) - _u/_v 

def MaxKL(_V,_p,_c): 
    A = {"V": _V,"p":_p,"c":_c,"Z_hat": None}
    def f(_v,_args):
        a = 0
        b = 0
        for i in _args["Z_hat"]:
            a += _args["p"][i]*np.log(_v-_args["V"][i])
            b += _args["p"][i]/(_v-_args["V"][i]) 
        return a+np.log(b)-_args["c"]

    def f_prime(_v,_args):
        a = 0
        b = 0
        c = 0
        for i in _args["Z_hat"]:
            a += _args["p"][i]/(_v-_args["V"][i])
            b += _args["p"][i]*(_v-_args["V"][i])**2
            c += _args["p"][i]*(_v-_args["V"][i])
        return a - (b/c)

    # Solving for Z
    q = np.zeros(_p.shape[0])
    q_hat = np.zeros(len(_p))
    Z_zero = np.where(_p == 0)
    Z_hat = np.where(_p > 0)
    A["Z_hat"] = Z_hat
    I_star = []
    np.argmax(_V)
    for i in I_star:
        if i == np.argmax(_V):
            I_star.append(i)

    # Solving for r, v and some q* values
    belowBounds = False
    for i in I_star:
        if f(_V[i],) < 0:
            belowBounds = True
            v = _V[i]
            return
    if belowBounds:
        r = 1 - np.exp(f(v,_V,_p,Z_hat))
        for i in I_star:
            q[i] = r/len(I_star) # Come back?
        for i in Z_zero:
            if i not in I_star:
                q[i] = 0
    else:
        r = 0
        for i in Z_zero:
            q[i] = 0
        v = newtons_method(f,f_prime,A) 

    #Solving for q*
    for i in Z_zero:
        q_hat[i] = _p[i]/(v-_V[i])
    
    q_hat_zero_sum = np.sum(q_hat)
    for i in Z_hat:
        q[i] = ((1-r)*(_p[i]/(v-_V[i])))/q_hat_zero_sum


def newtons_method(func,dfunc,_args = None,x0 = 0,a =.001,N = 1000,divExept = .001):
    xNext = x0
    for _ in range(N):
        x = xNext
        fx = func(x,_args)
        dfx = dfunc(x,_args)
        if dfx == 0 or dfx ==None:
            dfx = dfunc(x+divExept,_args)
        xNext = x-fx/dfx
        if np.abs(x-xNext) < a:
            return x 
    return x
