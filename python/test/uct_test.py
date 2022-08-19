import gym
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from solvers.uct import UCT
from select_action import actions as act
from optimizers.optimization import Bellman

env = gym.make('Taxi-v3')
env.reset()
print(env.action_space)
actionSelectionSelection = act.action_selection(act.UCB1,{"c":.93}) 
actionSelectionRollout = act.action_selection(act.randomAction)

solverUCT = UCT(Bellman(env,0),env,actionSelectionSelection,actionSelectionRollout)

solverUCT.render_ = False
solverUCT.search()
print(solverUCT.tree_[4].a_[0].s_prime_i_)