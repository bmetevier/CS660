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
    d0s = np.array([0.1, 0.2, 0.5])
    sigmas = 0.1 * np.ones(params["n_arms"])

    # collect data
    data = []
    for d0 in d0s:
        means = np.array([1, 0])
        print(f"trying delta0={d0}")
        params['delta0'] = d0
        data.append(run_bandit(params, means, sigmas ** 2, experiment=1))

    fname = "UCB_exp1"
    ytitle = 'attack cost'
    xtitle = 'round'
    labels = ['delta0' + str(i) + '=' + str(d0s[i]) for i in range(d0s.size)]
    plot(data, xtitle, ytitle, labels, params["n_rounds"], fname, "", "semilogx")

def UCB_experiment_2(params):
    print("running UCB experiment 2...")
    # set up arm reward distributions
    sigmas = np.array([0.5, 0.2, 0.1])
    params['delta0'] = 0.1

    # collect data
    data = []
    for sigma in sigmas:
        means = np.array([1, 0])
        print(f"trying sigma={sigma}")
        ss = sigma * np.ones(params["n_arms"])
        data.append(run_bandit(params, means, ss ** 2, experiment=1))

    fname = "UCB_exp2"
    ytitle = 'attack cost'
    xtitle = 'round'
    labels = ['sigma' + str(i) + '=' + str(sigmas[i]) for i in range(sigmas.size)]
    plot(data, xtitle, ytitle, labels, params["n_rounds"], fname, "", "semilogx")

def UCB_experiment_3(params):
    variances = 0.1 * np.ones(2)
    means = np.array([1, 0])
    params['delta0'] = 0.1
    params['n_jobs'] = None
    # collect data
    attacks = [True, False]
    data = []
    for attack in attacks:
        print("attacking") if attack else print("not attacking")
        params["attack"] = attack
        data.append(run_bandit(params, means, variances, experiment=3))

    # data plotting info
    fname = "UCB_exp3"
    ytitle = 'target arm pulls'
    xtitle = 'round'
    labels = ['attacked', 'without attack']
    plot(data, xtitle, ytitle, labels, params["n_rounds"], fname)

#if __name__ == '__main__':
#    #RUN UCB EXPERIMENTS
#    n_arms = 2
#    delta=0.05
#    n_rounds = 10
#    n_trials = 5
#    target_arm = 1
#    n_jobs = 7
#    algo = "ucb"
#    
#    params = {"n_arms":n_arms, "delta":delta, 
#              "n_rounds":n_rounds, 
#              "n_trials":n_trials, "target":target_arm, 
#              "n_jobs": n_jobs, "algo": algo}
#              
#    UCB_experiment_1(params)
