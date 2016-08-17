import numpy as np

def flipLabels(Y,p):
    Yn=np.copy(Y)
    n=np.size(Yn)
    n_flips=int(n*p)
    I=np.random.permutation(n)
    sel=I[0:n_flips]
    Yn[sel]=-1*Yn[sel]
    return Yn