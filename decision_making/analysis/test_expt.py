import pandas as pd 

def test(params: dict):
    if "a" in params["values"]:
        print("A ", params["values"]["a"])
        val = {"name": "a", "value": params["values"]["a"]}
    if "b" in params["values"]:
        print("B ", params["values"]["b"])
        val = {"pseudonym": "b", "value": params["values"]["b"]}
    return pd.DataFrame([val])

    ## Save the param to a pandas series and return it
