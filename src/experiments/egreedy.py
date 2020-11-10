import numpy as np
from IPython import embed
from plotter import plot
from utils import run_bandit

#############
#EGREEDY EXPERIMENTS
#############
def EG_experiment_1(params):
    """Vary the mean of non-target arm."""
    
    print("running egreedy experiment 1...")
    #set up arm reward distributions
#    mus = np.array([1, 2, 5])
    mus = np.array([5])
    target_mu=0
    sigmas = 0.1*np.ones(params["n_arms"])
    
    #collect data
    data = []
    for mu in mus:
        means = np.array([mu, target_mu])
        print(f"trying mu={mu}")
        data.append(run_bandit(params, means, sigmas**2))
        
    #data plotting info
    fname = "egreedy_exp1"
    ytitle = 'attack cost'
    xtitle = 'log(round)'
    labels = ['mu='+str(mus[i]) for i in range(mus.size)]
    plot(data, xtitle, ytitle, labels, params["n_rounds"], fname, "logx")
        
    return data

def EG_experiment_2(params):
    """Vary the variance of the non-target arm."""
    print("running egreedy experiment 2...")
    #set up arm reward distributions
    mus = np.array([1, 0])
    target_stdev=1
    stdevs = np.array([0.05, 0.1, 0.2])
#    stdevs = np.array([0.2])
    
    #collect data
    data = []
    for stdev in stdevs:
        sigmas = np.array([stdev, target_stdev])
        print(f"trying sigma={stdev}")
        data.append(run_bandit(params, mus, sigmas**2))
        
    #data plotting info
    fname = "egreedy_exp2"
    ytitle = 'log(attack cost)'
    xtitle = 'loglog(round)'
    labels = ['sig='+str(stdevs[i]) for i in range(stdevs.size)]
    plot(data, xtitle, ytitle, labels, params["n_rounds"], fname, "loglogxlogy")
        
    return data

def EG_experiment_3(params):
    mus = np.array([1, 0])
    target_stdev=1
    stdevs = np.array([0.05, 0.1, 0.2])
    
    
    #data plotting info
    fname = "egreedy_exp3"
    ytitle = 'target arm pulls'
    xtitle = 'round'
    labels = ['attacked', 'without attack']
    plot(data, xtitle, ytitle, labels, params["n_rounds"], fname, "logx")

if __name__ == '__main__':
    #RUN EGREEDY EXPERIMENTS
    n_arms = 2
    delta=0.025
    n_rounds = 5
    n_trials = 10#**2#3
    target_arm = 1
    n_jobs = None
    algo = "egreedy"
    
    params = {"n_arms":n_arms, "delta":delta, 
              "n_rounds":n_rounds, 
              "n_trials":n_trials, "target":target_arm, 
              "n_jobs": n_jobs, "algo": algo}
              
    EG_experiment_1(params)