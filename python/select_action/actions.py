import math
import numpy as np
import random

class action_selection():
    """
    Selects which actions to take through a function
    Description: Class that defines which action is selected. 
    """
    def __init__(self, _func, _const : list = []) -> None:
        """
        Constructor
        Args:
            Self (action_selection): action selection object
            _func (funciton): function for actions to be selected
            _const (list): List of constants to use for function
        Returns:
            actions_selection: Action Selection Object
        """
        self.func_ : function = _func
        self.const_ : list = _const
        pass


    def return_action(self,_s,_a,_param = []):
        return self.func_(_s,_a, self.const_,_param)


def UCB1(_s,_a,_const,_param=[]):
    UCB = math.nan
    optAction = math.nan
    random.shuffle(_a)
    for a in _a:
        aVal =_param[a]["Q"] + _const["c"]*np.sqrt(((2*np.log(_s.t_))/_param[a]["N"])) #UCB1 Equation
        if aVal > UCB or np.isnan(UCB):
            UCB = aVal
            optAction = a
    return optAction


def randomAction(_s,_a,_const,_param):
    return np.random.randint(len(_a))