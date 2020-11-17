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
        
    def manipulate_reward(self, reward, action, bandit, sigmas):
        """
        Performs the reward manipulation attack on the Egreedy bandit.
        
        Arguments:
            reward (float): reward environment provides the bandit
            bandit (EGreedyBandit): bob the bandit
            variances (np.ndarray (bandit.n_arms,)): variances of reward 
                distribution associated with each arm
        """
        invalid_means = -np.inf in bandit.means
        if invalid_means:
            raise ValueError("bandit has not updated all arms")
            
        attack_bandit = not (bandit.explore or action==self.target)
        
        if attack_bandit:
            # if bandit.n_arm_pulls.sum() == 2:
            #     print(max(0, self._get_alpha(reward, bandit, sigmas[bandit.action])))
            self._alpha = max(0, self._get_alpha(reward, bandit, sigmas[action]))
        else:
            self._alpha = 0
        return reward-self.alpha
    
    def _get_alpha(self, reward, bandit, sigma):
        """
        Arguments:
            reward (float): environment reward
            bandit (Bandit): egreedy bandit 
            sigma (float): stdev of reward dist associated with the 
                            most recently sampled arm
        """
        
        #func isn't called when action=target arm (total arms = 2)
        action = 0
        Ni = bandit.n_arm_pulls[action]
        Ni_mui = bandit.means[action] * Ni + reward
        beta = self._get_beta(bandit.n_arm_pulls[self.target], sigma, bandit.n_arms)
        Ni_muK = (Ni + 1) * (bandit.means[self.target] - 2 * beta)
        return Ni_mui - Ni_muK
        
    def _get_beta(self, N, sigma, n_arms):
        """
        compute the confidence interval 
        
        Arguments:
            N (int): number of arm pulls of arm i up to round t
            sigma (float): stdev of reward distribution associated with arm i
        """
        outer = (2 * sigma) / N
        inner = (np.pi ** 2) * (n_arms * N ** 2) / (3 * self.delta)
        return np.sqrt(outer * np.log(inner))