import numpy as np
from .agent import Agent

class UCB(Agent):
    """
    Emulates the EGreedy algorithm described in 'Adversarial Attacks
    Against Multi-Armed Bandits'
    """
    def __init__(self, n_arms):
        """
        """
        
        super().__init__()
        if n_arms<2:
            raise ValueError("number of arms must be greater than 1")
            
        pass
        
    def sample_action(self, action=None):
        """chooses to explore or exploit, then 
        samples an action in the environment"""
        
        pass
        
    def update_means(self, reward):
       pass