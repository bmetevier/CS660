import numpy as np
from IPython import embed

from .attacker import MABAttacker

class UCBAttacker(MABAttacker):
    def __init__(self, target, delta0):
        super().__init__()
        self.alphas = None
        self.delta0 = delta0
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
        if self.alphas is None:
            self.alphas = np.zeros([bandit.n_arms])

        attack_bandit = not (bandit.explore or bandit.action == self.target)
        if attack_bandit:
            self._alpha = 0
        else:
            self._alpha = 0
        return reward-self.alpha

    def _get_alpha(self, reward, bandit, sigma):
        return None

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