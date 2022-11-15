import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from select_action import action_selection

act_sel_funcs = {
    "ambiguity_aware"   : action_selection.ambiguity_aware,
    "ucb1"              : action_selection.UCB1,
    "mcgs_dm"           : action_selection.mcgs_dm,
    "mcgs_best_action"  : action_selection.mcgs_best_action,
    "random_action"     : action_selection.random_action
}