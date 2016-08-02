import numpy as np
import matplotlib.pyplot as plt
from spectral_reg_toolbox import flipLabels

def linear_data(N, pflip,  m=np.random.random() * 2 + 0.01, b=np.random.random()*0.5, s=0.1):
    #Sample a dataset from a linear separable dataset
    #   [X, Y, m, b, s] = linear(N, m, b)
    #    INPUT 
    #    N      1x2 vector that fix the numberof samples from each class
    #    m      slope of the separating line. Default is random.    
    #    b      bias of the line. Default is random.
    #    s      standard deviation of the gaussian noise. Default is 0.1
    #    OUTPUT
    #    X data matrix with a sample for each row 
    #       Y vector with the labels
    #
    #   EXAMPLE:
    #       [X, Y] = linearData([10, 10]);
    
    X=np.zeros((1,2))
    while(np.size(X,axis=0) < N[1]):
        xx = np.random.random(1)
        yy = np.random.random(1)
        fy = xx * m + b
        if(yy <= fy):
            Xi=np.concatenate((np.add(xx,(np.random.standard_normal((1,1))*s)),np.add(yy,(np.random.standard_normal((1,1))*s))), axis=1)
            X = np.concatenate((X,Xi))
 
    while(np.size(X,axis=0) < np.sum(N)):
        xx = np.random.random(1)
        yy = np.random.random(1)
        fy = xx * m + b
        if(yy > fy):
            Xi=np.concatenate((np.add(xx,(np.random.standard_normal((1,1))*s)),np.add(yy,(np.random.standard_normal((1,1))*s))), axis=1)
            X = np.concatenate((X,Xi))
    
    X[0,:]=np.concatenate((np.add(xx,(np.random.standard_normal((1,1))*s)),np.add(yy,(np.random.standard_normal((1,1))*s))), axis=1)
    Y = np.ones((np.sum(N), 1))
    Y[0:N[0]]  = -1
    Y=flipLabels(Y,pflip)
    return X, Y