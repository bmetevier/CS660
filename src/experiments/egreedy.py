import numpy as np
from IPython import embed
from plotter import plot
from utils import run_bandit
import os, sys

#############
#EGREEDY EXPERIMENTS
#############
def EG_experiment_1(params, plot=False, repid=0, swarm=False):
    """Vary the mean of non-target arm."""
    
    print("running egreedy experiment 1...")
    #set up arm reward distributions

    mus = np.array([1, 2, 5])
    sigmas = 0.1*np.ones(params["n_arms"])
    
    #collect data
    data = []
    for mu in mus:
        means = np.array([mu, 0])
        print(f"trying mu={mu}")
        data.append(run_bandit(params, means, sigmas**2, experiment=1))

    if swarm:
        if not os.path.exists("/mnt/nfs/scratch1/ktakatsu/egreedy_exp1"):
            os.makedirs("/mnt/nfs/scratch1/ktakatsu/egreedy_exp1")
        with open('/mnt/nfs/scratch1/ktakatsu/egreedy_exp1/{}.npy'.format(str(repid)), 'wb') as f:
            np.save(f, np.array(data))

    #data plotting info
    if plot:
        fname = "egreedy_exp1"
        ytitle = 'attack cost'
        xtitle = 'log(round)'
        labels = ['mu='+str(mus[i]) for i in range(mus.size)]
        plot(data, xtitle, ytitle, labels, params["n_rounds"], fname, "logx")
    return data

def EG_experiment_2(params, plot=False, repid=0, swarm=False):
    """Vary the variance of the non-target arm."""
    print("running egreedy experiment 2...")
    #set up arm reward distributions
    mus = np.array([1, 0])
    target_stdev=1
    stdevs = np.array([0.05, 0.1, 0.2])
    #collect data
    data = []
    for stdev in stdevs:
        sigmas = np.array([stdev, target_stdev])
        print(f"trying sigma={stdev}")
        data.append(run_bandit(params, mus, sigmas**2, experiment=2))

    if swarm:
        if not os.path.exists("/mnt/nfs/scratch1/ktakatsu/egreedy_exp2"):
            os.makedirs("/mnt/nfs/scratch1/ktakatsu/egreedy_exp2")
        with open('/mnt/nfs/scratch1/ktakatsu/egreedy_exp2/{}.npy'.format(str(repid)), 'wb') as f:
            np.save(f, np.array(data))

    #data plotting info
    if plot:
        fname = "egreedy_exp2"
        ytitle = 'log(attack cost)'
        xtitle = 'loglog(round)'
        labels = ['sig='+str(stdevs[i]) for i in range(stdevs.size)]
        plot(data, xtitle, ytitle, labels, params["n_rounds"], fname, "loglogxlogy")
        
    return data

def EG_experiment_3(params, plot=False, repid=0, swarm=False):
    variances = 0.1 * np.ones(2)
    means = np.array([1,0])

    #collect data
    attacks = [True, False]
    data = []
    for attack in attacks:
        print("attacking") if attack else print("not attacking")
        params["attack"] = attack
        data.append(run_bandit(params, means, variances, experiment=3))
    if swarm:
        if not os.path.exists("/mnt/nfs/scratch1/ktakatsu/egreedy_exp3"):
            os.makedirs("/mnt/nfs/scratch1/ktakatsu/egreedy_exp3")
        with open('/mnt/nfs/scratch1/ktakatsu/egreedy_exp3/{}.npy'.format(str(repid)), 'wb') as f:
            np.save(f, np.array(data))

    if plot:
        #data plotting info
        fname = "egreedy_exp3"
        ytitle = 'target arm pulls'
        xtitle = 'round'
        labels = ['attacked', 'without attack']
        plot(data, xtitle, ytitle, labels, params["n_rounds"], fname)

if __name__ == '__main__':
    task_id = sys.argv[1]
    #RUN EGREEDY EXPERIMENTS
    n_arms = 2
    delta=0.025
    n_rounds = 10**5
    n_trials = 1
    target_arm = 1
    n_jobs = None
    algo = "egreedy"
    params = {"n_arms":n_arms, "delta":delta, 
              "n_rounds":n_rounds, 
              "n_trials":n_trials, "target":target_arm, 
              "n_jobs": n_jobs, "algo": algo}

    EG_experiment_1(params, repid=task_id, swarm=True)
    EG_experiment_2(params, repid=task_id, swarm=True)
    EG_experiment_3(params, repid=task_id, swarm=True)
    # EG_experiment_3(params)
