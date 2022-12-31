#!/usr/bin/python
"""
This script is samples a set of variables and adds them to a configuration file
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os

from numpy import var
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import json
import logging

from numpy import random

import nestifydict as nd

class SampleExperiment():
    """
    Samples variables for experiments
    
    Default location for files is "<parent>/config/"
    
    *Later I may consider making a verion that can be parse within an experimental run, this would be conducive to filtering but would require a change to specification*
    Initial idea is to add to n_trials loop, then filters can be applied to each element. Additionally sampled variables can be added with specifiers and removed from combination generation
    
    :param sample_file: (str) Filename for sample params
    :param save_file: (str) Filename for saving params
    :param log_level: (str) Log level (does not override default values), *default*: WARNING

    """
    def __init__(self, sample_file : str, save_file : str, log_level : str = "WARNING"):
        
        super(SampleExperiment, self).__init__()
        
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        self._log_level = log_levels[log_level]
                                             
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=self._log_level)
        self._log = logging.getLogger(__name__)
        
        self._log.warn("Initializing Sample Expt from " + sample_file + " to " + save_file)
        
        f = open(parent + "/config/" + sample_file +  ".json")
        sample_config = json.load(f)
        self._sample_config = sample_config["samples"]
        self._n_samples = sample_config["n_samples"]
        print(self._sample_config)
        f.close
        
        f._save_file = save_file

    def sample(self):
        """
        Samples variables from config and saves them to save
        """
        f = open(parent + "/config/" + self._save_file +  ".json")
        data = json.load(f)
        
        for el in self._sample_config:
            if isinstance(el["target"], str) and el["target"] == "all":
                el["target"] = range(len(data))
            elif not isinstance(el["target"], list):
                el["target"] = [el["target"]]
            
            for target in el["target"]:
                key = nd.find_key(data[target], el["var"])
                
                if "choice" in el:
                    num = el["num"] if "num" in el else 1
                    vals = self.sample_discrete(el["choice"], num)
                else:
                    num = el["num"] if "num" in el else 1
                    inc = el["increment"] if "increment" in el else None
                    vals = self.sample_continuous(el["lower"], el["upper"], num, inc)

                nd.recursive_set(data["target"], key, vals)
    
        json.dump(data, f)
        f.close()
            
    def sample_continuous(self, lower, upper, num, inc = None):
        
        if inc != None:
            pass        
        else:
            pass
    
    def sample_discrete(self, choices, num):
        
        pass
    
    def filter(self):
        """
        Not Implemented... Perhaps add in later version
        """
        pass

if __name__=='__main__':
    sample_config_file = sys.argv[1]
    save_config_file = sys.argv[2]
    if len(sys.argv) >= 4:
        log_level = sys.argv[3]
    else:
        log_level = "WARNING"
    
    sampler = SampleExperiment(sample_config_file, save_config_file, log_level)
    
    sampler.sample()