from math import gamma
import gym
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from planners.uct import UCT
from decision_making.select_action import action_selection as act
from optimizers.optimization import Bellman




env = gym.make('Taxi')
env_sim = gym.make('Taxi')
print(env.action_space)
actionSelectionSelection = act.action_selection(act.UCB1,{"c":.75}) 
actionSelectionRollout = act.action_selection(act.randomAction)
solverUCT = UCT(env,env_sim,[0,1,2,3,4,5],actionSelectionSelection,actionSelectionRollout)
solverUCT.render_ = False
solverUCT.seed = 5
while(True):
    solverUCT.playGame()

print(solverUCT.tree_[0].V_)
print(solverUCT.tree_[solverUCT.n_vertices_-1].N_)