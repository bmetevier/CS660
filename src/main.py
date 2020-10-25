from experiments.egreedy import *
from experiments.ucb import *

"""
Re-Implementation of experiments for ADVERSARIAL ATTACKS IN STOCHASTIC BANDITS
#http://papers.nips.cc/paper/7622-adversarial-attacks-on-stochastic-bandits.pdf
"""

def main():
    
    #RUN EGREEDY EXPERIMENTS
    n_arms = 2
    delta=0.025
    n_rounds = 10**3#5
    n_trials = 10**2#3
    target_arm = 1
    n_jobs = 7
    
    params = {"n_arms":n_arms, "delta":delta, 
              "n_rounds":n_rounds, 
              "n_trials":n_trials, "target":target_arm, 
              "n_jobs": n_jobs}
              
    EG_experiment_2(params)
    
    #RUN UCB EXPERIMENTS
    
if __name__ == '__main__':
    main()