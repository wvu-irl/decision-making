#!/usr/bin/python

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

"""_summary_ This file is intended to run a user specified
        decision making algorithm from those available in config
        
    
"""

## CONFIG -------------------------------------------------

algo = sys.argv[1]
env_name = sys.argv[2]
do_render = sys.argv[3]
if len(sys.argv) >= 5:
    fp = sys.argv[4]
else:
    fp = None

#load JSON
    pass
    #default filepath    print("saving output to ")
    
print("Testing ", algo, " in ", env_name)
print("Rendering? ", do_render)
if fp != None:
    print("Saving output to ", fp)



## Evaluation -------------------------------------------------



## Post Processing---------------------------------------------