
import numpy as np
import copy

arr2d = np.zeros([40,40])
my_d = {"pose": [1,2], "arr": arr2d}
print(my_d)
d2 = copy.deepcopy(my_d)
my_d["arr"][0][1] = 1
print(my_d)
print(d2)