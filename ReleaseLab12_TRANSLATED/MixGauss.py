'''
Created on 21. jun. 2016

@author: Sigurd Lekve
'''
import numpy as np
import matplotlib.pyplot as plt

def MixGauss(means, sigmas, n):
    #
    # function MixGauss(means, sigmas, n)
    #
    # means: (size dxp) and should be of the form [m1, ... ,mp] (each mi is
    # d-dimensional
    #
    # sigmas: (size px1) should be in the form [sigma_1;...; sigma_p]  
    #
    # n: number of points per class
    #
    # X: obtained input data matrix (size 2n x d) 
    # Y: obtained output data vector (size 2n)
    #
    # EXAMPLE: MixGauss([[[0],[0]],[[1],[1]]],[[0.5],[0.25]],100); 
    # generates a 2D dataset with two classes, the first one centered on (0,0)
    # with variance 0.5, the second one centered on (1,1) with variance 0.25. 
    # each class will contain 100 points
    #
    # to visualize: plt.scatter(X[:,0],X[:,1],75,Y,edgecolor='None')
    d=np.size(means, axis=0)
    p=np.size(means, axis=1)

    X = np.zeros((n*p,d))
    Y = np.zeros((n*p,1))
    for i in range(0,p):
        m = means[:][i]
        S = sigmas[i]
        Xi = np.zeros((n,d))
        Yi = np.zeros((n,1))
        for j in range(0,n):
            x = S*np.random.standard_normal((d,1)) + m
            x=np.transpose(x)
            Xi[j,:] = x[:]
            Yi[j] = i+1
        if i==0:
            X=Xi
            Y=Yi
        else:
            X=np.concatenate((X,Xi))
            Y=np.concatenate((Y,Yi))
    return X,Y