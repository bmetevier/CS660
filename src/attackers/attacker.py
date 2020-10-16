from abc import ABC, abstractmethod

class MABAttacker(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def manipulate_reward(self, env_reward, bandit):
        """employs a designated reward attack strategy
        
        Returns:
            reward (float): the manipulated reward intended to trick the 
                            bandit into pulling the target arm
            """
        pass
