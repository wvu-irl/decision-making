

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

path = current + "../"
file = "true_test.csv"

with open(path + file, 'r') as f:
    df = pd.read_csv(f)
    
    for index, row in df.iterrows():
        dist = row["distribution"]
        
        entropy = 0
        for key in dist:
            entropy += dist[key] * np.log(dist[key])
            
        entropy = -entropy
        
        df.at[index, "entropy"] = entropy
        
    df.to_csv(path + "true_test_entropy.csv")
