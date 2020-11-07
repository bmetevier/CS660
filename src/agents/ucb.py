import numpy as np
from .agent import Agent

class UCB(Agent):
    """
    Emulates the EGreedy algorithm described in 'Adversarial Attacks
    Against Multi-Armed Bandits'
    """
    def __init__(self, n_arms, sigmas):
        """
        """
        super().__init__()
        if n_arms<2:
            raise ValueError("number of arms must be greater than 1")
            
        self.round = 0
        self._means = -np.inf * np.ones(n_arms)  # start pessimistic
        self._action = None

        self.n_arm_pulls = np.zeros(n_arms)
        self._explore = True
        self.n_arms = n_arms
        self.sigmas = sigmas

    @property
    def explore(self):
        """True if in the most recent round the bandit explored"""
        return self._explore

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def action(self):
        return self._action

    @property
    def means(self):
        """estimated mean reward for each arm"""
        return self._means

    def sample_action(self, action=None):
        """chooses to explore or exploit, then 
        samples an action in the environment"""
        self.round += 1
        if action == None:
            self._explore = False
            score = self.means + 3*np.sqrt(self.sigmas)*np.sqrt(np.log(self.round)/self.n_arm_pulls)
            action = np.argmax(score)

        self._action = action
        self.n_arm_pulls[action] += 1
        return action
        
    def update_means(self, reward):

        if self.means[self.action] == -np.inf:
            self.means[self.action] = reward
        else:
            previous_sum = self.means[self.action] * (self.n_arm_pulls[self.action] - 1)
            self.means[self.action] = (previous_sum + reward) / self.n_arm_pulls[self.action]