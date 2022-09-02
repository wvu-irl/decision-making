import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from select_action.ambiguity_toolbox import *

n = 100
for i in range(n):
    print(1-(i+1)/n, get_accuracy((i+1)/n,125,0.005))
#generate_invA(3)