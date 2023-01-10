#!/usr/bin/python
"""
This script is intended to set default, visualize, and update decision making configuration.
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import json

def update( var, val, update_all : bool = False, core_file : str = current + "/core.json",):
    """
    Update one or more config values

    :param var: () parameter to update
    :param val: () new value of parameter
    :param update_all: (bool) if true, accepts var as a list of keys, *defualt*: False
    :param core_file: (str) location of core to overwrite, *default*: current + "/core.json"
    """
    with (core_file, 'r+') as f:
        data = json.load(f)
        if update_all:
            for key, v in zip(var, val):
                data[key] = v  
        else:
            data[var] = val
        json.dump(data)  
        
def reset_core(core_file : str = current + "/core.json", default_file : str = current + "/core_default.json"):
    """
    Resets core configuration to defaults

    :param core_file: (str) location of core to overwrite, *default*: current + "/core.json"
    :param default_file: (str) location of default file to overwrite core, *default*: current + "/core_default.json"
    """
    with open(default_file, "rb") as default_f:
        params = json.load(default_f)
    with open(core_file, "w+") as core_f:
        json.dump(params, core_f, indent = 4)
        
def initialize_core(path : str = current):
    """
    Initializes core variables from path

    :param path: (str) where to set initial values
    """
    base_file = path + "core_default_base.json"
    default_file = path + "core_default.json"
    core_file = path + "core.json"
    
    with open(base_file, "rb") as default_f:
        params = json.load(default_f)
        
    for el in params.keys():
        if isinstance(el,str) and "file" in el:
            params[el] = path + params[el]
            
    with open(default_file, "w+") as default_f:
        json.dump(params, default_f, indent = 4)
    
    reset_core(core_file, default_file)
          
def print_config(core_file : str= current + "/core.json"): 
    """
    Prints configuration file settings
    
    :param file_name: (str) Location of configuration params, *default*: current + "/core.json"
    """
    print("Getting Decision Making Toolbox Configuration")
    with open(core_file, "rb") as f:
        params = json.load(f)
        len_key = 0
        for key in params.keys():
            if len(str(key)) > len_key:
                len_key = len(str(key))
        len_key += 2
        for key in params.keys():
            print(f'{key:-<{len_key}} -> ' + str(params[key]))     

if __name__=='__main__':
    file_name = current + "/core.json"
    print(file_name)
    default_file_name = current + "/core_default.json"
    
    if sys.argv[1] == "default":
        reset_core(file_name, default_file_name)
        
    elif sys.argv[1] == "update":
        i = 2
        while i <= len(sys.argv) -2:
            update(sys.argv[i], sys.argv[i+1], False, file_name)
            i += 2
            
    elif sys.argv[1] == "init":
        initialize_core(current)
        
    elif sys.argv[1] == "print" or len(sys.argv) < 2:
        print_config(file_name)
        
    else:
        print("Invalid Configuation Directive")
