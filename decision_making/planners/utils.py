#!/usr/bin/python

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from planners import *
from select_action.utils import * 

def get_agent(alg_params=None, env_params=None):
    if alg_params["alg"] == "aogs":
        act_sel = action_selection(act_sel_funcs[alg_params["action_selection"]["function"]], alg_params["action_selection"]["params"])
        return aogs.AOGS(act_sel, alg_params, env_params)#env, act_sel, alg_config["max_iter"], env_config["reward_bounds"], [alg_config["epsilon"], alg_config["delta"]], alg_config["gamma"])
        #     #aogs = AOGS(env, act_select, _performance = [0.1, 0.05], _bounds = bounds)

    # elif alg_params["alg"] == "gbop":
    #     return gbop.GBOP(env, act_select, _performance = [0.1, 0.05], _bounds = bounds)
    
    # elif alg_params["alg"] == "uct":
    #     return uct.UCT(env, act_select, _performance = [0.1, 0.05], _bounds = bounds)
    