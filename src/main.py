from experiments.egreedy import *
from experiments.ucb import *

"""
Re-Implementation of experiments for ADVERSARIAL ATTACKS IN STOCHASTIC BANDITS
#http://papers.nips.cc/paper/7622-adversarial-attacks-on-stochastic-bandits.pdf
"""

def main():
    
    #RUN EGREEDY EXPERIMENTS
    n_arms = 2
    delta = 0.05
    n_rounds = 10**5#5
    n_trials = 10**2/2#3
    target_arm = 1
    n_jobs = 7
    algo = "UCB" # {"UCB", "egreedy"}
    delta0 = 0.1 # parameter for UCB
    
    params = {"n_arms":n_arms, "delta":delta, "delta0":delta0,
              "n_rounds":n_rounds, 
              "n_trials":n_trials, "target":target_arm,
              "n_jobs": n_jobs, "algo": algo}
              
    # EG_experiment_1(params)
    UCB_experiment_1(params)


    #RUN UCB EXPERIMENTS
    
if __name__ == '__main__':
    main()