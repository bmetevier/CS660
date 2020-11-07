import numpy as np
from IPython import embed
from plotter import plot
from utils import run_bandit

#############
#UCB EXPERIMENTS
#############

def UCB_experiment_1(params):
    print("running UCB experiment 1...")
    # set up arm reward distributions
    mus = np.array([5])
    target_mu = 0
    sigmas = 0.1 * np.ones(params["n_arms"])

    # collect data
    data = []
    for mu in mus:
        means = np.array([mu, target_mu])
        print(f"trying mu={mu}")
        data.append(run_bandit(params, means, sigmas ** 2))

def UCB_experiment_2(params):
    pass
def UCB_experiment_3(params):
    pass