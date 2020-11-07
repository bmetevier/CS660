import numpy as np
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from IPython import embed

from env import Environment
from agents import EGreedy, UCB
from attackers import EGreedyAttacker, UCBAttacker

def _run_bandit(n_trials, params, means, variances, data):
    """
    Performs the bandit->environment->attacker interaction for a single trial.
    
    Returns:
        data (np.ndarray (n_rounds,)): attack cost per round
    """
    for trial in range(int(n_trials)):
        env = Environment(means, variances)
        attacker, bandit = get_alice_and_bob(params, variances)
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

def get_alice_and_bob(params, variances):
    assert params["algo"] in {"egreedy", "UCB"}, "Incorrect algorithm name"
    if params["algo"] == "egreedy":
        bandit = EGreedy(params["n_arms"])
        attacker = EGreedyAttacker(params["target"], params["delta"])
    elif params["algo"] == "UCB":
        bandit = UCB(params["n_arms"], variances)
        attacker = UCBAttacker(params["target"], params["delta"], params["delta0"])
    return attacker, bandit

class Manager(BaseManager):
    #https://stackoverflow.com/questions/25938187/trying-to-use-multiprocessing-to-fill-an-array-in-python
    pass

def run_bandit(params, means, variances):
    """
    Performs the bandit->environment->attacker interaction for a number of trials. 
    
    Arguments:
        params (Python dict): holds parameter values of interest (see main.py)
        means (np.ndarray (n_arms,)): mean of the reward distribution associated with each arm
        variances (np.ndarray (n_arms,)): stdv of the reward distribution associated with each arm
    
    Returns:
        data (np.ndarray (n_rounds,)): mean attack cost per round averaged over N trials
    """
    
    def get_trials_per_worker():
        n_workers = params["n_jobs"]
        #distribute trials evenly among workers
        distribution = (params["n_trials"]//n_workers)*np.ones(n_workers)
        #remainder trials are added to the last worker
        distribution[-1] += int(params["n_trials"]%n_workers)
        return distribution
    
    if params["n_jobs"]==1:
        data = np.zeros(params["n_rounds"])
        _run_bandit(params["n_trials"], params, means, variances, data)
    else:
        trial_dist = get_trials_per_worker()
        data = _get_data_array(params["n_rounds"])
        
        jobs = []
        for w in range(params["n_jobs"]):
            p = mp.Process(target=_run_bandit, 
                           args=(trial_dist[w], params, means, variances, data))
            jobs.append(p)
            p.start() 
        for process in jobs:
            process.join() 
    return np.array(data)/params["n_trials"]

def _get_data_array(n_rounds):
    Manager.register('np_zeros', np.zeros, mp.managers.ArrayProxy)
    m = Manager()
    m.start()
    return m.np_zeros(n_rounds)

def _printlog(trial):
    if trial%20==0: print(f"trial {trial}")