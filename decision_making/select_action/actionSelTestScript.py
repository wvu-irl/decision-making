import decision_making.select_action.action_selection as act
import numpy as np

trials : int = 10000
trueVal : list[float] = [7,5.8,6,4,6.7]
actionID : list[int] = [i for i in range(len(trueVal))]
param : list[dict] = [{"Q":10, "N":1 } for i in range(len(trueVal)) ]
const : dict = {"c": 2.5}

actionSelect : act.action_selection = act.action_selection(act.UCB1,const)

for i in range(trials):
    bestAction : int = actionSelect.return_action(i,[],actionID,param)
    param[bestAction]["N"] += 1
    param[bestAction]["Q"] += (1/param[bestAction]["N"])*((np.random.randn(1)*2+trueVal[bestAction])-param[bestAction]["Q"])
    print(param)