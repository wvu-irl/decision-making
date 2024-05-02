import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from select_action import action_selection
from select_action import mm_action_selection
from select_action import model_selection

act_sel_funcs = {
    "ambiguity_aware"   : action_selection.ambiguity_aware,
    "ucb1"              : action_selection.UCB1,
    "gbop_dm"           : action_selection.gbop_dm,
    "gbop_best_action"  : action_selection.gbop_best_action,
    "random_action"     : action_selection.random_action,
    "mm_ambiguity_aware": mm_action_selection.ambiguity_aware
}

model_sel_funcs = {
    "progressive_widening"  : model_selection.progressive_widening
}
