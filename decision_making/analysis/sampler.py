#!/usr/bin/python
"""
This script is samples a set of variables and adds them to a configuration file
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

from copy import deepcopy
import sys
import os

from numpy import var
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import json

import numpy as np
from numpy import random

import nestifydict as nd

__all__ = ["from_file_to_file", "sample", "sample_all"]

def from_file_to_file(sample_file : str, save_file : str):
    """
    Samples variables for experiments
    
    :param sample_file: (str) Filename for sample params
    :param save_file: (str) Filename for saving params
    """
    f_sample = open(sample_file)
    f_save = open(save_file)
    
    sample_config = json.load(f_sample)
    save_output = json.load(f_save)
    
    f_sample.close()
    
    f_save.dump(sample_all(sample_config, save_output))
    f_save.close()
    
def sample_all(config : dict, output : dict):
    """
    Sample configuration to an output dictionary
    
    Sampling uses `NestifyDict <https://pypi.org/project/nestifydict/>`_ so variables can be specified as their deepest key assuming this variable is only used in one place. 
    Otherwise the variable should be defined as a list.
    Nonexistent keys are skipped as the sampling declarations are contained here. 
    
    :param config: (dict) Sample configuration
    :param output: (dict) Where to add samples
    """
    d = {"output": output, "config": config}
    for el in config:
        key = nd.find_key(output,el)
        
        # Finds and overwrites params specified by other variables
        params = nd.recursive_get(d,key)
        for param in params:
            if isinstance(params[param],list):
                d[key + params] = deepcopy(nd.recursive_get(d,nd.find_key(params[param])))
        
        s = sample(key, d)
        if s is not None:     
            nd.recursive_set(output,["output"] + key[1:len(key)],s)
    
def sample(sample_params : dict):
    """
    Samples variables using specification
    
    **Note** For those using `from_file_to_file` or `sample_all`,
    sample params can be specified with the string key of another variable and it will be replaced with a copy of that value.
    Additionally, samples can be multi-dimensional
    
    :params sample_params: (dict) Contains sample parameters as follows
        
        *Continous sample*
        {
            "low": lower limit
            "high": upper limit
            "num_increments": (optional) number of increments to down sample a continuous space
            "num": (optional) number of times to sample
        }
        
        *Discrete sample*
        {
            "choice": Options to sample from
            "probability": probability to sample from
            "num": (optional) number of times to sample
        }
    
    :returns: (list) sampled values
    """
    rng = random.default_rng()
    
    if "choice" in sample_params:
        return sample_discrete(rng, sample_params)
    elif "low" in sample_params:
        return sample_continuous(rng, sample_params)
    return None
        
def sample_continuous(rng, params):
    """
    Samples one or more continuous distributions
    
    :param rng: (rng) random number generator
    :param params: (dict) params to sample dist of the form
        {
            "low": lower limit
            "high": upper limit
            "num_increments": (optional) number of increments to down sample a continuous space
            "num": (optional) number of times to sample
        }
    :return: list of samples
    """       
    if "num_increments" in params:
        temp = {}
        if "num" in params:
            temp["num"] = params["num"]
        temp["choice"] = np.linspace(params["low"], params["high"], params["num_increments"])
        return sample_discrete(rng,temp) 
    else:
        num = params["el"] if "num" in params else None
        vals = rng.random(np.shape(params["low"]),num)
        return (np.asarray(params["high"])-np.asarray(params["low"]))*vals + np.asarray(params["low"])
    
def sample_discrete(rng, params):
    """
    Samples one or more discrete distributions
    
    :param rng: (rng) random number generator
    :param params: (dict) params to sample dist of the form
        {
            "choice": Options to sample from
            "probability": probability to sample from
            "num": (optional) number of times to sample
        }
    :return: list of samples
    """
    num = params["el"] if "num" in params else None
    p = params["probability"] if "probability" in params else None
    return rng.choice(params["choice"], num, replace = False, p = p)
    
def filter(self):
    """
    Not Implemented... Perhaps add in later version
    """
    pass

if __name__=='__main__':
    sample_config_file = sys.argv[1]
    save_config_file = sys.argv[2]
    
    sample_config_file = parent + "/config/" + sample_config_file +  ".json"
    save_config_file = parent + "/config/" + save_config_file +  ".json"

    from_file_to_file(sample_config_file, save_config_file)
    