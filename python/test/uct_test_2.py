from math import gamma
import gym
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from solvers.uct import UCT
from select_action import actions as act
from gym_envs.gridworld import GridWorld



dim = [40,40]
goal = [10,10]
p = 0.05
env = GridWorld(dim, goal, p)
env_sim = GridWorld(dim, goal, p)


actionSelectionSelection = act.action_selection(act.UCB1,{"c":10}) 
actionSelectionRollout = act.action_selection(act.randomAction)

solverUCT = UCT(env,env_sim,env.a_,actionSelectionSelection,actionSelectionRollout)
solverUCT.render_ = True
solverUCT.seed = 5

env_sim.render()
s=env_sim.get_observation()
while(True):
    solverUCT.reinnit(s)
    a = solverUCT.learn(s)
    print("act " + str(a.a_))
    env_sim.reset(s)
    s, r,d,info = env_sim.step(a.a_)
    # print("ss ",s)
    env_sim.render()
    print(r)
print(solverUCT.tree_[0].V_)
print(solverUCT.tree_[solverUCT.n_vertices_-1].N_)