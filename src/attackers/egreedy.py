import numpy as np
from IPython import embed

from .attacker import MABAttacker

class EGreedyAttacker(MABAttacker):
    def __init__(self, target, delta=0.5):
        """
        Arguments:
            target (int): the target arm (arm the attacker wants 
                   the bandit to pull as often as possible).
            delta (float (0,1)): 
        """
        super().__init__()
        self.delta = delta
        self._alpha = None
        self.target = target
        
    @property
    def alpha(self):
        return self._alpha
        
    def manipulate_reward(self, reward, bandit, sigmas):
        """
        Performs the reward manipulation attack on the Egreedy bandit.
        
        Arguments:
            reward (float): reward environment provides the bandit
            bandit (EGreedyBandit): bob the bandit
            sigmas (np.ndarray (bandit.n_arms,)): stdevs of reward 
                distribution associated with each arm
        """
        invalid_means = -np.inf in bandit.means
        if invalid_means:
            raise ValueError("bandit has not updated all arms")
            
        attack_bandit = not (bandit.explore or bandit.action==self.target)
        if attack_bandit:
            self._alpha = max(0, self._get_alpha(reward, bandit, sigmas[bandit.action]))
            return reward-self.alpha
        else:
            self._alpha = 0
            return reward            
    
    def _get_alpha(self, reward, bandit, sigma):
        N_prev = bandit.n_arm_pulls[bandit.action]            
#        N_prev = N_curr-1
        prev_reward_sum = bandit.means[bandit.action] * N_prev
        N_target = bandit.n_arm_pulls[self.target]
        
        beta = self._get_beta(N_target, sigma, bandit.n_arms)
#        curr_mean = (prev_reward_sum + reward)/N_curr
#        constraint = curr_mean - 2*beta

        return (prev_reward_sum + reward - 
                (bandit.means[self.target]-2*beta)*(N_prev+1))
        
    def _get_beta(self, N, sigma, n_arms):
        """
        compute the confidence interval 
        
        Arguments:
            N (int): number of arm pulls of arm i up to round t
            sigma (float): stdev of reward distribution associated with arm i
        """
        outer = (2*sigma)/N
        inner = (np.pi**2)*(n_arms*N**2)/(3*self.delta)
        return np.sqrt(outer*np.log(inner))