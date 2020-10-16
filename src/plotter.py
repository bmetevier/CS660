import matplotlib.pyplot as plt
import numpy as np

def plot(ys, xtitle, ytitle, labels, n_rounds, logmsg=[]):
    
    X,Ys = get_axes(ys, n_rounds, logmsg)
    
    f = plt.figure()
    colors = ['black', 'grey', 'brown']
    for i in range(len(ys)):
        plt.plot(X, Ys[i], label=labels[i], color=colors[i])
    
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend()
    plt.show()
    f.savefig("plot.pdf", bbox_inches='tight')


#TODO: something better than this
def get_axes(ys,n_rounds,logmsg=""):
    if "logxlogx" in logmsg:
        pass
    elif "logx" in logmsg:
        X = np.round(np.logspace(1, 4, 50, base=10, endpoint=False)).astype(int)
    else:
        X = np.arange(n_rounds)
    
    if "logy" in logmsg:
        pass
    else:
        Ys = [y[X] for y in ys]
    
    return X, Ys