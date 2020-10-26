import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

def plot(ys, xtitle, ytitle, labels, n_rounds, filename, logmsg=""):
    
    X,Ys = get_axes(ys, n_rounds, logmsg)
    f = plt.figure()
    colors = ['black', 'grey', 'brown']
    for i in range(len(ys)):
        plt.plot(X, Ys[i], label=labels[i], color=colors[i])
    
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend()
    plt.show()
    f.savefig("experiments/results/"+filename+".pdf", bbox_inches='tight')

def get_axes(ys, n_rounds, logmsg=""):
    X = np.arange(1, n_rounds+1)
    if "loglogx" in logmsg:
        X = np.log(np.log(X))
    elif "logx" in logmsg:
        X = np.log(X)
    
    if "logy" in logmsg:
        #first two indices are zero
        Ys = [np.log(y[2:]) for y in ys]
        X = X[2:]
    else:
        Ys = ys

    return X, Ys