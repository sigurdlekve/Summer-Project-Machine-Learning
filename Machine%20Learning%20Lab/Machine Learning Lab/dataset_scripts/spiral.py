import numpy as np
import matplotlib.pyplot as plt

def spiral(N, s=0.35, wrappings = 2.5,  m=0.7):
    #Sample a dataset from a dataset separated by a sinusoidal line
    #   [X, Y, s, wrappings, m] = spiral(N, s, wrappings, m)
    #    INPUT 
    #    N         1x2 vector that fix the numberof samples from each class
    #    s         standard deviation of the gaussian noise. Default is 0.5.
    #    wrappings number of wrappings of each spiral. Default is random.
    #    m       multiplier m of x * sin(m * x) for the second spiral. Default is random.
    #    OUTPUT
    #    X data matrix with a sample for each row 
    #       Y vector with the labels
    #
    #   EXAMPLE:
    #       [X, Y] = spiral([10, 10])
    
    X = []
    oneDSampling = np.random.random((N[0], 1))*wrappings*np.pi
    X1var = np.concatenate((np.multiply(oneDSampling,np.cos(oneDSampling)), np.multiply(oneDSampling,np.sin(oneDSampling))), axis=1)
    X1 = np.add(X1var, np.random.standard_normal((N[0],2))*s)
    
    oneDSampling = np.random.random((N[1], 1))*wrappings*np.pi
    X2var=np.concatenate((np.multiply(oneDSampling,np.cos(m*oneDSampling)), np.multiply(oneDSampling,np.sin(m*oneDSampling))), axis=1)
    X2 = np.add(X2var, np.random.standard_normal((N[1],2))*s)   
    
    X=np.concatenate((X1,X2))
    
    Y = np.ones((np.sum(N), 1))
    Y[0:N[0]]  = -1
    return X, Y
