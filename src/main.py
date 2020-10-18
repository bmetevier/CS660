import numpy as np
from IPython import embed
from plotter import plot
from utils import run_bandit

"""
Re-Implementation of experiments for ADVERSARIAL ATTACKS IN STOCHASTIC BANDITS
#http://papers.nips.cc/paper/7622-adversarial-attacks-on-stochastic-bandits.pdf
"""

"""
#############
#EGREEDY EXPERIMENTS
#############
"""
def EG_experiment_1(params):
    """Vary the mean of non-target arm."""
    
    #set up arm reward distributions
    mus = np.array([1, 2, 5])
    target_mu=0
    sigmas = 0.1*np.ones(params["n_arms"])
    
    #collect data
    data = []
    for mu in mus:
        means = np.array([mu, target_mu])
        print(f"trying mu={mu}")
        data.append(run_bandit(params, means, sigmas**2))
        
    #data plotting info
    ytitle = 'attack cost'
    xtitle = 'round'
    labels = ['mu'+str(i)+'='+str(mus[i]) for i in range(mus.size)]
    plot(data, xtitle, ytitle, labels, params["n_rounds"], "logx")
        
    return data

def EG_experiment_2(params):
    pass
def EG_experiment_3(params):
    pass

"""
#############
#UCB EXPERIMENTS
#############
"""
def UCB_experiment_1(params):
    pass
def UCB_experiment_2(params):
    pass
def UCB_experiment_3(params):
    pass


def main():
    
    #RUN EGREEDY EXPERIMENTS
    n_arms = 2
    delta=0.025
    n_rounds = 10**4#5
    n_trials = 10**2#3
    target_arm = 1
    n_jobs = 7
    
    params = {"n_arms":n_arms, "delta":delta, 
              "n_rounds":n_rounds, 
              "n_trials":n_trials, "target":target_arm, 
              "n_jobs": n_jobs}
              
    EG_experiment_1(params)
    
    #RUN UCB EXPERIMENTS
    
if __name__ == '__main__':
    main()