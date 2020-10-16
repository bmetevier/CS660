import numpy as np
from env import Environment
from agents import EGreedy
from attackers import EGreedyAttacker

def run_bandit(params, means, variances, printlog=True):
    """
    Performs the bandit->environment->attacker interaction for some number
    of trials.
    
    Arguments:
        params (Python dict): holds parameter values of interest (see main.py)
        means (np.ndarray (n_arms,)): mean of the reward distribution associated with each arm
        variances (np.ndarray (n_arms,)): stdv of the reward distribution associated with each arm
    
    Returns:
        data (np.ndarray (n_rounds,)): mean attack cost per round averaged over N trials
    """
    
    data = np.zeros(params["n_rounds"])
    for trial in range(params["n_trials"]):
        if printlog: _printlog(trial)
        
        env = Environment(means, variances)
        bandit = EGreedy(params["n_arms"])
        attacker = EGreedyAttacker(params["target"], params["delta"])
        attack_cost = 0
        
        for r in range(params["n_rounds"]):
            initial_pull = r<params["n_arms"]
            if initial_pull: #pull each arm at least once
                action = bandit.sample_action(r)
                reward = env.get_reward(action)
            else:
                action = bandit.sample_action()
                env_reward = env.get_reward(action)
                reward = attacker.manipulate_reward(env_reward, bandit, variances)
                attack_cost+=attacker.alpha
                
            bandit.update_means(reward)
            
            data[r] += attack_cost
            
    return data/params["n_trials"]

def _printlog(trial):
    if trial%20==0: print(f"trial {trial}")