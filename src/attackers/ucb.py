import numpy as np
from IPython import embed

from .attacker import MABAttacker

class UCBAttacker(MABAttacker):
    def __init__(self, target, delta, delta0):
        super().__init__()
        self.alphas = None
        self.mu0 = None # store sample means before attacks
        self.delta = delta
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
        if self.mu0 is None:
            self.mu0 = bandit.means

        # print(bandit.n_arm_pulls)
        # print(self.mu0)
        # print(bandit.means)
        prev = self.mu0[bandit.action] * (bandit.n_arm_pulls[bandit.action]-1)
        self.mu0[bandit.action] = (prev + reward) / bandit.n_arm_pulls[bandit.action]

        attack_bandit = not (bandit.explore or bandit.action == self.target)
        if attack_bandit:
            self._alpha = max(0, self._get_alpha(reward, bandit, sigmas[bandit.action]))
            self.alphas[bandit.action] += self._alpha
        else:
            self._alpha = 0
        return reward-self.alpha

    def _get_alpha(self, reward, bandit, sigma):
        Ni = bandit.n_arm_pulls[bandit.action]
        Ni_mui = self.mu0[bandit.action] * (Ni-1) + reward
        beta = self._get_beta(bandit.n_arm_pulls[self.target], sigma, bandit.n_arms)
        Ni_muK = Ni * (bandit.means[self.target]-2*beta -self.delta0)

        return Ni_mui - self.alphas[bandit.action] - Ni_muK

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