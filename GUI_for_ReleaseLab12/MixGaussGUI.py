'''
Created on 21. jun. 2016

@author: Sigurd Lekve
'''
import numpy as np
import matplotlib.pyplot as plt
from Tkinter import *
from flipLabels import flipLabels

def MixGaussGUI_tr(means, sigmas, ntr, pflip):
    d=np.size(means, axis=0)
    p=np.size(means, axis=1)

    Xtr = np.zeros((ntr*p,d))
    Ytr = np.zeros((ntr*p,1))
    for i in range(0,p):
        m = means[:][i]
        S = sigmas[i]
        Xtr_i = np.zeros((ntr,d))
        Ytr_i = np.zeros((ntr,1))
        for j in range(0,ntr):
            x = S*np.random.standard_normal((d,1)) + m
            x=np.transpose(x)
            Xtr_i[j,:] = x[:]
            Ytr_i[j] = i+1
        if i==0:
            Xtr=Xtr_i
            Ytr=Ytr_i
        else:
            Xtr=np.concatenate((Xtr,Xtr_i))
            Ytr=np.concatenate((Ytr,Ytr_i))
    
    Ytr=flipLabels(Ytr,pflip)
    plt.figure()
    plt.scatter(Xtr[:,0],Xtr[:,1],50,Ytr,edgecolor='None')
    return Xtr,Ytr

#Test
#a=[[0],[0]]
#b=[[1],[1]]
#ab=[a,b]
#c=[[0.5],[0.3]]
#n=100
#X,Y=MixGauss(ab,c,n)
#plt.scatter(X[:,0],X[:,1],75,Y,edgecolor='None')
#plt.show()

def MixGaussGUI_ts(means, sigmas, nts, pflip):
    d=np.size(means, axis=0)
    p=np.size(means, axis=1)

    Xts = np.zeros((nts*p,d))
    Yts = np.zeros((nts*p,1))
    for i in range(0,p):
        m = means[:][i]
        S = sigmas[i]
        Xts_i = np.zeros((nts,d))
        Yts_i = np.zeros((nts,1))
        for j in range(0,nts):
            x = S*np.random.standard_normal((d,1)) + m
            x=np.transpose(x)
            Xts_i[j,:] = x[:]
            Yts_i[j] = i+1
        if i==0:
            Xts=Xts_i
            Yts=Yts_i
        else:
            Xts=np.concatenate((Xts,Xts_i))
            Yts=np.concatenate((Yts,Yts_i))
    
    Yts=flipLabels(Yts,pflip)
    plt.figure()
    plt.scatter(Xts[:,0],Xts[:,1],50,Yts,edgecolor='None')
    return Xts,Yts

def MixGaussGUI(ntr, nts, pflip, means, sigmas):
    Xtr, Ytr = MixGaussGUI_tr(means, sigmas, ntr)
    Xts, Yts = MixGaussGUI_ts(means, sigmas, nts)
    plt.show()
    return Xtr, Ytr, Xts, Yts
