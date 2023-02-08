import pandas as pd 
import time

def test(params: dict):
    my_str = ""
    for i in range(100000):
        my_str += "a"
        # print(my_str)
    if "a" in params["values"]:
        print("A ", params["values"]["a"])
        val = {"name": "a", "value": params["values"]["a"]}
    if "b" in params["values"]:
        print("B ", params["values"]["b"])
        val = {"pseudonym": "b", "value": params["values"]["b"]}
    return pd.DataFrame([val])

    ## Save the param to a pandas series and return it
