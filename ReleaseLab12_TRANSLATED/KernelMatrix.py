'''
Created on 21. jun. 2016

@author: Sigurd Lekve
'''
import numpy as np
from numpy import transpose as trsp
from SquareDist import SquareDist


def KernelMatrix(X1, X2, kernel, param):
    # Usage: K = KernelMatrix(X1, X2, kernel, param)
    # X1 and X2 are the two collections of points on which to compute the Gram matrix
    #
    # kernel: can be 'linear', 'polynomial' or 'gaussian'
    # param: is [] for the linear kernel, the exponent of the polynomial
    # kernel, or the variance for the gaussian kernel
    #
    
    if np.size(kernel) == 0:
        kernel = 'linear'
    if np.array_equal(kernel, 'linear'):
        K = np.dot(X1,trsp(X2))
    elif np.array_equal(kernel, 'polynomial'):
        K = np.power((1 + np.dot(X1,trsp(X2))),param)
    elif np.array_equal(kernel, 'gaussian'):
        K = np.exp((float(-1)/(float(2*param**2)))*SquareDist(X1,X2)) 
    return K