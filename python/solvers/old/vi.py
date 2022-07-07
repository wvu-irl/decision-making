import numpy as np

# this is from persepctive of adversary

# need functions for get_state_dim
    #then use ravel/unravel to map in and out

class VI():
    """
    Performs value iteration for the adversarial grid world problem
    Description: Ideally this would have been more general but wrote value iteration after making the env
    User defines:
        transition: this is an ndarray of the transition model
        reward: this is an ndarray of the reward model
    """

    def __init__(self, _opt, _epsilon, _gamma):
        """
        Constructor, initializes MDP model

        Args:
            self (VI_Adversary): Object to initialize
            _epsilon (double): convergence criteria
            _gamma (double): Discount factor
            _num_actions (int): number of actions

        Returns:
            VI_Adversary: Q-value iteration object
        """
        super(VI, self).__init__()
        self.model_ = _opt.problem_
        self.epsilon_ = _epsilon
        self.gamma_ = _gamma
        self.state_dim_ = _opt.problem_.get_num_states()
        self.n_ = 1
        for el in self.state_dim_:
            self.n_ *= el
        self.Q_ = np.zeros([self.n_,_opt.problem_.get_num_actions()])
        self.opt_ = _opt
        # print(self.world_bounds)      

    def solve(self, _Q = None):
        """
        Performs value iteration on the model

        Args:
            self (VI_Adversary): Object to initialize
            _transition (ndarray): transition model
            _reward (ndarray): reward model
            _num_actions (int): number of actions
        """
        if _Q != None:
            self.Q_ = _Q

        diff = self.epsilon_*10
        while diff > self.epsilon_:
            temp = self.Q_.copy()
            self.update_Q()
            diff = np.sum(np.absolute(np.absolute(self.Q_) - np.absolute(temp)))
            print("Q-diff ", diff)
        return self.Q_

    def get_Q(self):
        return self.Q_

    def update_Q(self):
        temp = self.Q_.copy()
        for s in range(self.n_):
            sm = np.unravel_index(s, self.state_dim_)
            # print(sm)
            # print(self.model_.get_actions(sm))
            # print(s)
            for a in self.model_.get_actions(sm):
                temp[s][a] = self.opt_.compute_Q(sm, a, self.gamma_, self.Q_)

        self.Q_ = temp.copy()

    def get_policy(self, _s):
        s = np.ravel_multi_index(_s, self.state_dim_)
        #print("Q ", self.Q_[s])
        return np.argmax(self.Q_[s])