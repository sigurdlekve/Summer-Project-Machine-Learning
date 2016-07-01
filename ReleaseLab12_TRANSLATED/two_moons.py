'''
Created on 29. jun. 2016

@author: Sigurd Lekve
'''
import numpy as np
import scipy.io as sio
from flipLabels import flipLabels

def two_moons(npoints, pflip):
    
    moons_dataset=sio.loadmat('moons_dataset.mat')
    Xtr=moons_dataset['Xtr']
    Ytr=moons_dataset['Ytr']
    Xts=moons_dataset['Xts']
    Yts=moons_dataset['Yts']
    
    npoints=np.min([100, npoints])
    I=np.random.permutation(100)
    sel=I[1:npoints]
    Xtr=Xtr[sel,:]
    Ytrn=flipLabels(Ytr[sel],pflip)
    Ytsn=flipLabels(Yts,pflip)
    return Xtr, Ytrn, Xts, Ytsn
