import numpy as np
from IPython import embed

from .attacker import MABAttacker

class UCBAttacker(MABAttacker):
    def __init__(self):
        super().__init__()
        pass
        
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
        pass       