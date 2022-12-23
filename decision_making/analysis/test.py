from copy import deepcopy

def pack_dict(d_in, d_structure):
    """
    Repackages list of keys and values into a nested dictionary (consumes input dictionary)

    :param d_in: 1d dict containing values
    :param d_structure: (dict) dictionary containing structure and default values
    :return: Restructured dictionary
    """
    d_out = deepcopy(d_structure)
    for key in d_structure:
        if type(d_structure[key]) == dict:
            d_out[key] = pack_dict(d_in, d_structure[key])
        elif key in d_in:
            d_out[key] = deepcopy(d_in[key])
            d_in.pop(key)

    return deepcopy(d_out)

a = {"b":1, "c": 2, "d": 3, "e": 4}
g = {"b":[], "f": {"c":[], "h":{"d": [], "e": []}}}

k = pack_dict(a,g)

print(a)
print(k)
print(g)