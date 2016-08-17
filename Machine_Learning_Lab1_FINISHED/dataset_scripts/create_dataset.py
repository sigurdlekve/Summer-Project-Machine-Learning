import math
import numpy as np
import matplotlib.pyplot as plt

from gaussian import gaussian
from linear_data import linear_data
from sinusoidal import sinusoidal
from moons import moons
from spiral import spiral

def create_dataset(N, data_type, noise,
                    sm=0.1, d=1, angle=3*np.pi,
                    ndist=1, means=np.array([[-0.25, 0.0],[0.25, 0.0]]), sigmas=np.array([0.1, 0.1]),
                    ml=1, b=0.2, sl=0.05,
                    ssin=0.02,
                    sspi=1, wrappings = 4,  mspi=0.5):
    #Sample a dataset from different distributions
    #   X, Y = create_dataset(N, type, noise)
    #
    #   INPUT 
    #       N          Number of samples
    #       data_type  Type of distribution used. It must be one from 
    #                  'Moons' 'Gaussian' 'Linear' 'Sinusoidal' 'Spiral'
    #       noise      probability to have a wrong label in the dataset.
    #    
    #       'Moons' parameters:
    #           1- sm: standard deviation of the gaussian noise. Default is 0.1
    #           2- d: 1X2 translation vector between the two classes. With d = 0
    #                 the classes are placed on a circle. Default is 1.
    #           3- angle: rotation angle of the moons in (radians). Default is 3*np.pi.
    #
    #       'Gaussian' parameters:
    #           1- ndist: number of gaussians for each class. Default is 1.    
    #           2- means: vector of size(2*ndist X 2) with the means of each gaussian. 
    #              Default is np.array([[-0.25, 0.0],[0.25, 0.0]]).
    #           3- sigmas: A sequence of covariance matrices of size (2*ndist, 2). 
    #              Default is np.array([0.1, 0.1]).
    #
    #       'Linear' parameters:
    #           1- ml: slope of the separating line. Default is 1.    
    #           2- b: bias of the line. Default is 0.2.
    #           3- sl: standard deviation of the gaussian noise. Default is 0.05.
    #
    #       'Sinusoidal' parameters:
    #           1- ssin: standard deviation of the gaussian noise. Default is 0.02.
    #
    #       'Spiral' parameters:
    #           1- sspi: standard deviation of the gaussian noise. Default is 1.
    #           2- wrappings: wrappings number of wrappings of each spiral. Default is 4.
    #           3- mspi: multiplier m of x * sin(m * x) for the second spiral. Default is 0.5.
    #
    #  OUTPUT
    #   X data matrix with a sample for each row 
    #   Y vector with the labels
    #
    #   EXAMPLE:
    #       X, Y = create_dataset(100, 'SPIRAL', 0.01)
    #       X, Y = create_dataset(100, 'SPIRAL', 0, sspi=0.1, wrappings=2, m=2);
    
    NN = [int(math.floor(float(N) / 2.0)), int(math.ceil(float(N) / 2.0))]
    
    if data_type=='Moons':
        X, Y = moons(NN, sm, angle, d)
    elif data_type=='Gaussian':
        X, Y = gaussian(NN, ndist, means, sigmas)
    elif data_type=='Linear':
        X, Y = linear_data(NN, ml, b, sl)
    elif data_type=='Sinusoidal':
        X, Y = sinusoidal(NN, ssin)
    elif data_type=='Spiral':
        X, Y = spiral(NN, sspi, wrappings, mspi)
    else:
        print 'Specified dataset type is not correct. It must be one of "Moons", "Gaussian", "Linear", "Sinusoidal", "Spiral"'
    
    swap=np.random.random((np.size(Y, axis=0), np.size(Y, axis=1)))<=noise
    Y[swap]=Y[swap]*-1
    
    return X, Y
