

## this file imports a csv
# for each row, it calculates the entropy for a dictionary in one of the columns
# it then writes the entropy to a new column in the csv

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pandas as pd
import numpy as np

from ast import literal_eval

path = current + "/../data/"
file = "true_expt_all.csv"

with open(path + file, 'r') as f:
    df = pd.read_csv(f)
    
    for index, row in df.iterrows():
        #entropy
        dist = literal_eval(row["distribution"])
        
        entropy = 0
        for el in dist: 
            entropy += el[1] * np.log(el[1])
            
        entropy = -entropy
        
        df.at[index, "entropy"] = entropy
        
        #ratio of objects
        num_o = row["num_objects"]
        num_r = row["num_returned"]
        
        df.at[index, "returned_ratio"] = num_r / num_o
        
        # df.at[index, "failed_drop_ratio"] = row["num_failed_drops"] / num_o
        # df.at[index, "failed_grab_ratio"] = row["num_failed_grabs"] / num_o
        
        
    df.to_csv(path + "true_test_post_proc.csv")