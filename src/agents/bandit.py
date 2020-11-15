from .agent import Agent
import numpy as np
from IPython import embed

class Bandit(Agent):
    def __init__(self, bandit_name, n_arms, variances=None):
        super().__init__()
        self.round = 0
        self._means = -np.inf * np.ones(n_arms) #start pessimistic
        self._action = None
        
        self.n_arm_pulls = np.zeros(n_arms)
        self._explore = False
        self.n_arms = n_arms
    
        self._ucb = False
        if bandit_name=="UCB":
            self.sigmas = np.sqrt(variances)
            self._ucb = True 
        
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
        
        if action==None:
            if self._ucb:
                score = self.means + 3*self.sigmas*np.sqrt(np.log(self.round+1)/self.n_arm_pulls)
                action = np.argmax(score)
            else:
                if self._will_explore():
                    action = np.random.randint(self.n_arms)
                else: #exploit
                    #return ties randomly (np.argmin returns first min)
                    argmaxes = np.flatnonzero(self.means==self.means.max())
                    action = np.random.choice(argmaxes)
        return action
    
    def _will_explore(self):
        r = np.random.uniform()
        self._explore = r < 1/(self.round+1)
        return self._explore
    
    def update_params(self, action, reward):
        """Updates the agent's internal parameters"""
        self.round +=1 
        self._action = action
        self.n_arm_pulls[action] += 1
        self._update_means(reward)
    
    def _update_means(self, reward):
        """Updates the agent's internal mean estimations of each arm"""
        if self.means[self.action] == -np.inf:
            self.means[self.action] = reward
        else:
            previous_sum = self.means[self.action] * (self.n_arm_pulls[self.action] - 1)
            self.means[self.action] = (previous_sum + reward) / self.n_arm_pulls[self.action]
