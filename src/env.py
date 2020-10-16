import numpy as np
    
class Environment(object):
    
    def __init__(self, means:np.ndarray, variances:np.ndarray):
        """
        
        Arguments:
            means (np.ndarray (K,)):
                the means of each of the sigma-sub-gaussian distribution
                from which each reward from K arms is drawn
            variances (np.ndarray (K,)):
                the sigma values of the sigma-sub-gaussian distribution 
                from which each reward is drawn
        """
        if (means.size != variances.size):
            raise ValueError("invalid mean and variance arrays")
        
        self.means = means
        self.vars = variances
    
    @property
    def n_actions(self):
        return self.means.size
    
    def get_reward(self, action):
        return np.random.normal(self.means[action], self.vars[action])        