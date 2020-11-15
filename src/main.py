from experiments.egreedy import *
from experiments.ucb import *
import sys
import os

"""
Re-Implementation of experiments for ADVERSARIAL ATTACKS IN STOCHASTIC BANDITS
#http://papers.nips.cc/paper/7622-adversarial-attacks-on-stochastic-bandits.pdf
"""

def main(task_id):

    if not os.path.exists("experiments/results"):
        os.makedirs("experiments/results")
    if not os.path.exists("experiments/results/data"):
        os.makedirs("experiments/results/data")

    #RUN EGREEDY EXPERIMENTS
    n_arms = 2
    delta = 0.05
    n_rounds = 10**2
    n_trials = 1#3

    target_arm = 1
    n_jobs = 6
    algo = "UCB"  # {"UCB", "egreedy"}
    delta0 = 0.1  # parameter for UCB
    
    params = {"n_arms":n_arms, "delta":delta, "delta0":delta0,
              "n_rounds":n_rounds, 
              "n_trials":n_trials, "target":target_arm,
              "n_jobs": n_jobs, "algo": algo}
              
    UCB_experiment_1(params, repid=task_id)
    UCB_experiment_2(params, repid=task_id)
    UCB_experiment_3(params, repid=task_id)


    #RUN UCB EXPERIMENTS
    
if __name__ == '__main__':
    task_id = sys.argv[1:]
    main(task_id)
