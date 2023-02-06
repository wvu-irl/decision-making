import pandas as pd 

def test(params: dict):
    if "a" in params["values"]:
        print("A ", params["values"]["a"])
    if "b" in params["values"]:
        print("B ", params["values"]["b"])

    ## Save the param to a pandas series and return it
    return pd.Series(params["values"]) 
