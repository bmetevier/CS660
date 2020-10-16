from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self):
        super().__init__()
        
    @property
    @abstractmethod
    def action(self):
        """the most recent action taken by the bandit"""
        pass
    
    @abstractmethod
    def sample_action(self):
        """Samples an action."""
        pass
    
    @abstractmethod
    def update_means(self):
        """Updates the agent's internal mean estimations of each arm"""
        pass


        