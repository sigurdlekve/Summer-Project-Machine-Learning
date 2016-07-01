'''
Created on 28. jun. 2016

@author: Sigurd Lekve
'''
import numpy as np

def flipLabels(Y,p):
    n=np.size(Y)
    n_flips=int(n*p)
    I=np.random.permutation(n)
    sel=I[0:n_flips]
    Y[sel]=-1*Y[sel]
    return Y