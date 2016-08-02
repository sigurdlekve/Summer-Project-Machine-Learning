import numpy as np
import matplotlib.pyplot as plt
from spectral_reg_toolbox import flipLabels

def gaussian(N, nflips, ndist=1, means=np.array([[0, 0],[1.5, 1.5]]), sigmas=np.array([0.6, 0.6])):
    #Sample a dataset from a mixture of gaussians
    #   [X, Y, ndist, means, sigmas] = gaussian(N, ndist, means, sigmas)
    #    INPUT 
    #    N      1x2 vector that fix the numberof samples from each class
    #    ndist  number of gaussian for each class. Default is 1.    
    #    means  vector of size(2*ndist X 2) with the means of each gaussian. 
    #           Default is random.
    #    sigmas A sequence of covariance matrices of size (2*ndist, 2). 
    #           Default is random.
    #    OUTPUT
    #    X data matrix with a sample for each row 
    #       Y vector with the labels
    #
    #   EXAMPLE:
    #       [X, Y] = gaussian([10, 10]);
    
    X = []
    for i in range(1,N[0]+1):
        dd = int(np.floor(np.random.random(1) * ndist))
        if X==[]:
            X = np.random.standard_normal((1,2)) * sigmas[0] + means[0, :]
            #X= np.random.standard_normal((1,2)) * sigmas[dd*2:dd*2+1, :] + means[dd, :]
        else:
            X1=np.random.standard_normal((1,2)) * sigmas[0] + means[0, :]
            #X1=np.random.standard_normal((1,2)) * sigmas[dd*2:dd*2+1, :] + means[dd, :]
            X=np.concatenate((X,X1))
         
    for i in range(1,N[1]+1):
        dd = int(np.floor(np.random.random(1) * ndist + ndist))
        X1 = np.random.standard_normal((1,2)) * sigmas[1] + means[1, :]
        X=np.concatenate((X,X1))
        
    
    Y = np.ones((np.sum(N), 1))
    Y[0:N[0]]  = -1
    Y=flipLabels(Y,nflips)
    return X, Y