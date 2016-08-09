import numpy as np
import matplotlib.pyplot as plt

def sinusoidal(N, s=0.02):
    #Sample a dataset from a dataset separated by a sinusoidal line
    #   [X, Y, s] = sinusoidal(N, m, b)
    #    INPUT 
    #    N      1x2 vector that fix the numberof samples from each class
    #     s      standard deviation of the gaussian noise. Default is 0.02
    #    OUTPUT
    #    X data matrix with a sample for each row 
    #       Y vector with the labels
    #
    #   EXAMPLE:
    #       [X, Y] = sinusoidal([10, 10])
    
    X=np.zeros((1,2))
    while (np.size(X, axis=0) < N[0]):
        xx = np.random.random(1)
        yy = np.random.random(1)
        fy = 0.7* 0.5 *(np.sin(2*np.pi*xx))+0.5
        if (yy <= fy):
            Xi=np.concatenate((np.add(xx,(np.random.standard_normal((1,1))*s)),np.add(yy,(np.random.standard_normal((1,1))*s))), axis=1)
            X = np.concatenate((X,Xi))
    
    while(np.size(X, axis=0) < np.sum(N)):
        xx = np.random.random(1)
        yy = np.random.random(1)
        fy = 0.7* 0.5 *(np.sin(2*np.pi*xx))+0.5
        if(yy > fy):
            Xi=np.concatenate((np.add(xx,(np.random.standard_normal((1,1))*s)),np.add(yy,(np.random.standard_normal((1,1))*s))), axis=1)
            X = np.concatenate((X,Xi))
    
    X[0,:]=np.concatenate((np.add(xx,(np.random.standard_normal((1,1))*s)),np.add(yy,(np.random.standard_normal((1,1))*s))), axis=1)
    Y = np.ones((np.sum(N), 1))
    Y[0:N[0]]  = -1
    return X, Y