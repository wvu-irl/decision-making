import numpy as np
import random
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

def lower_bound(_s, _a, _solver, _delta):
    #max r _ gamma*max_q q*U
    pass

def upper_bound(_s, _a, _solver, _delta):
    #min r _ gamma*min_q q*U
    pass

def beta(_n, _delta):
    pass

def KL_divergence(_p, _q):
    d = 0
    for p,q in zip(_p,_q):
        d += p *np.log(p/q)