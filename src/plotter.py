import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

def plot(ys, xtitle, ytitle, labels, n_rounds, filename, logmsg=""):
    
    X,Ys = get_axes(ys, n_rounds, logmsg)
    f = plt.figure()
    colors = ['black', 'grey', 'brown']
    for i in range(len(Ys)):
        if i<3:
            plt.plot(X, Ys[i], label=labels[i], color=colors[i])
        else: #dotted lines for y=1/2 and y=1
            plt.plot(X, Ys[i], "--", color="gray")

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend()
    plt.show()
    f.savefig("experiments/results/"+filename+".pdf", bbox_inches='tight')


def get_axes(ys, n_rounds, logmsg=""):
    X = np.arange(1, n_rounds+1)
    if "loglogx" in logmsg:
        X = np.log(np.log(X)[1:])
    elif "logx" in logmsg:
        X = np.log(X)
    
    if "logy" in logmsg:
        #first two indices are zero
        Ys = [np.log(y[2:]) for y in ys]
        X = X[1:]
        
        #add y=1/2 and y=1 
#        Ys.append(np.log(np.arange(Ys[0].size+1))[1:]) #m=1
#        Ys.append((np.log(np.arange(Ys[0].size+1))**2)[1:]) #m=1/2
    else:
        Ys = ys
    return X, Ys