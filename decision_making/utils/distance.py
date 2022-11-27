import numpy as np

def get_distance(s1, s2 = None, W = None, distance = "Eucilidean", params = None):
        
        if type(s2) != list:
            s2 = np.zeros(len(s1))
        if W == None:
            W = np.ones(len(s1))
        
        if distance == "Euclidean":
            distance = "L"
            params = 2
        elif distance == "Manhattan":
            distance = "L"
            params = 1
            
        if distance[0] == "L":
            if params == 0:
                total = 0
                for x1,x2,w in zip(s1,s2,W):
                    total += int(x1-x2 != 0)*W
                return total
            elif params == np.inf:
                max_diff = 0
                for x1,x2,w in zip(s1,s2,W):
                    if np.abs(x1-x2)*w > max_diff:
                        max_diff = np.abs(x1-x2)*w
                return max_diff
            else:
                total = 0
                for x1,x2,w in zip(s1,s2,W):
                    total += w* ((x1-x2)**params)
                return total**(1/params)
                
                
                
    