import matplotlib.pyplot as plt
import numpy as np

def plot(ys, xtitle, ytitle, labels, n_rounds, logmsg=[]):
    
#    X,Ys = get_axes(ys, n_rounds, logmsg)
    X = np.arange(1, n_rounds+1)
    f = plt.figure()
    colors = ['black', 'grey', 'brown']
    for i in range(len(ys)):
        plt.plot(np.log(X), ys[i], label=labels[i], color=colors[i])
    
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend()
    plt.show()
    f.savefig("plot.pdf", bbox_inches='tight')


##TODO: something better than this
#def get_axes(ys,n_rounds,logmsg=""):
#    if "logxlogx" in logmsg:
#        X = np.round(np.logspace(1, 3, 50, base=np.e, endpoint=False)).astype(int)
#    elif "logx" in logmsg:
##        X = np.round(np.logspace(1, 3, 50, base=np.e, endpoint=False)).astype(int)
#        X = np.arange(n_rounds)
#    else:
#        X = np.arange(1n_rounds)
#    
#    if "logy" in logmsg:
#        pass
#    else:
#        Ys = [y[X] for y in ys]
#    
#    return X, Ys