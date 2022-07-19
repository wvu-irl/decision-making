
from cmath import inf
from re import I
import numpy as np
import random

def get_avg(_dist):
    n = 0
    avg = 0
    for el in _dist:
        n += el[0]
        avg += el[0]*el[1]
    return avg, n

def get_ind(_dist, _n):
    p = 0
    i = 0
    while p < _n:
        p += _dist[i][0]
        i += 1
    return i   

def compute_bf_accuracy(_dist, _e):
    # iterate over all and check to see if +/-e is too close to 1/0

    # add elements for each pair and add 2/n mass
    pass 

def compute_discount_bf(_bf, c, _l, _u):
    pass

def compute_lower_expectation(_bf):
    pass

def compute_upper_expectation(_bf):
    pass

def get_action_epsilon_greedy(_actions, _e, _rng):
    n = _rng.random()
    if n < _e:
        Q_s = -inf
        ind = 0
        for i in range(len(_actions)):
            Q, num = get_avg(_actions[i])
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
        Q, n = get_avg(_actions[i])
        ucb = Q + _c*np.sqrt(np.log(N)/n)
        if ucb > ucb_max:
            ucb_max = ucb
            ind = i
    return ind

def get_action_ambiguity(_actions, _e, _alpha, _l, _u):
    exp_max = -inf
    ind = 0
    for i in range(len(_actions)):
        bf, n = compute_bf_accuracy(_actions[i])
        t = 0
        for a in _actions[i]: t += a[0]
        c = 1 - 2*n*np.exp(-8*t*(_e**2)/n)  ### NEED TO TRY ALSO WITH N+1!!!
        bf = compute_discount_bf(bf, c, _l, _u)
        low_exp = compute_lower_expectation(bf)
        up_exp = compute_upper_expectation(bf)
        exp = _alpha*low_exp + (1-_alpha)*up_exp
        if exp > exp_max:
            exp_max = exp
            ind = i
    return ind




## Initialize params
# Assume we are using epsilon-greedy
num_el = 10
num_trials = 1e3
num_iter = 1e3
num_a = 5
num_outcomes = 10
R = list(range(0,5,25))
rng = np.random.default_rng()

# e-greedy 
epsilon = [0]* num_el
for i in range(len(epsilon)): epsilon[i] = i/num_el

# ucb1
c = [0]* num_el
for i in range(len(epsilon)): epsilon[i] = i**2*0.5

# true distribution for each arm
alpha = [0]* num_el
for i in range(len(epsilon)): epsilon[i] = i/num_el


## Models
temp =[]
for el in R:
    temp.append((0,el))
dist = [temp.copy] * num_a

# True model 

# dim-1, action, dim-2 outcomes


# e-greedy model

# ucb1 model

# ambiguity model

## Data
e_greedy_r = np.zeros(num_trials,num_iter, num_el)
ucb_r = np.zeros(num_trials,num_iter, num_el)
amb_r = np.zeros(num_trials,num_iter, num_el)



##  Sample and select
for i in list(range(num_trials)):
    for j in list(range(num_iter)):
        # sample all actions

        # update models


## Plot

# avg rewards
