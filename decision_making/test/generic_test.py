#!/usr/bin/python

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import json 

from planners import *
from envs import *

"""_summary_ This file is intended to run a user specified
        decision making algorithm from those available in config
        
    
"""

## CONFIG -------------------------------------------------
algo = sys.argv[1]
alg_config_file = sys.argv[2]
env_config_file = sys.argv[3]

if len(sys.argv) >= 5:
    fp = sys.argv[4]
else:
    fp = None

f = open(current + "../config/algorithms/" + alg_config_file +  ".json")
alg_config = json.load(f)
f = open(current + "../config/envs/" + env_config_file +  ".json")
env_config = json.load(f)   

# SETUP CONFIG -------------------

# ENVS
if env_config == "grid world":
    pass
elif env_config == "sailing":
    pass
elif env_config == "grid trap":
    pass


# ALGS
if env_config == "grid world":
    pass
elif env_config == "sailing":
    pass
elif env_config == "grid trap":
    pass

# PRINT CONFIG -------------------
    
print("Testing ", algo, " in ", env_name)
print("Rendering? ", do_render)
if fp != None:
    print("Saving output to ", fp)



## Evaluation -------------------------------------------------



## Post Processing---------------------------------------------