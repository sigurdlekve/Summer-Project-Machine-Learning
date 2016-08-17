import numpy as np
import matplotlib.pyplot as plt

def moons(N, s=0.1, angle = 3 * np.pi, d=1):
    #Sample a dataset from two "moon" distributions 
    #   X, Y = moons(N, s, angle, d)
    #    INPUT 
    #    N     1x2 vector that fix the numberof samples from each class
    #    s     standard deviation of the gaussian noise. Default is 0.1
    #    angle rotation angle of the moons. Default is 3*pi.
    #    d     translation vector between the two classes. With d = 0
    #          the classes are placed on a circle. Default is 1.
    #    OUTPUT
    #    X data matrix with a sample for each row 
    #       Y vector with the labels
    #
    #   EXAMPLE:
    #       X, Y = moons([10, 10])
    
    d1 = -(np.random.random((1, 2)) * 0.6) + np.array([-0.2, -0.2])
    d2=np.array([[float(np.cos(-angle)), float(-np.sin(-angle))], [float(np.sin(-angle)), float(np.cos(-angle))]])
    d = (np.dot(d2,d1.T)).T
    
    oneDSampling =  np.ones((1,N[0]))*np.pi + np.random.random((1,N[0]))*1.3*np.pi + np.ones((1,N[0]))*angle
    X1 = np.concatenate((np.sin(oneDSampling.T), np.cos(oneDSampling.T)), axis=1) + np.random.standard_normal((N[0],2))*s
    
    oneDSampling =  np.random.random((1,N[1]))*1.3*np.pi + np.ones((1,N[1]))*angle
    X2 =np.concatenate((np.sin(oneDSampling.T), np.cos(oneDSampling.T)), axis=1) + np.random.standard_normal((N[1],2))*s + np.array(np.tile(d, (N[1], 1)))

    X=np.concatenate((X1, X2))
    Y = np.ones((np.sum(N), 1))
    Y[1:N[1]]  = -1
    return X, Y