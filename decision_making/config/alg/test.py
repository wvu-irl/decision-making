from reconfigurator import compiler
import json
import sys
import os 
from importlib.resources import path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
print(sys.path)
f = open('/home/projects/decision_making/decision-making/decision_making/config/alg/aogs_expt.json')
file_test = json.load(f)
X=compiler.compile_to_list(file_test)
print(X)
