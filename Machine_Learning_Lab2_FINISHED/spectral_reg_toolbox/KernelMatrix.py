import numpy as np
from numpy import transpose as trsp

def SquareDist(X1, X2):
    n = np.size(X1,axis=0)
    m = np.size(X2,axis=0)
    
    sq1 = np.sum(np.multiply(X1,X1),axis=1)
    sq2 = np.sum(np.multiply(X2,X2),axis=1)
    
    D1=np.dot(trsp(np.matrix(sq1)),np.ones((1,m)))
    D2=np.dot(np.ones((n,1)),np.matrix(sq2))
    D3=2*np.dot(X1,trsp(X2))
    D = D1+D2-D3
    return D   

def KernelMatrix(X1, X2, kernel_type, KerPar):
    #KERNEL Calculates a kernel matrix.
    #   K = KERNELMATRIX(KNL, KPAR, X1, X2) calculates the nxN kernel matrix given
    #   two matrix X1[n,d], X2[N,d] with kernel type specified by 'knl':
    #       'Linear'   - linear kernel, 'kpar' is not considered
    #       'Polynomial'   - polinomial kernel, where 'kpar' is the polinomial degree
    #       'Gaussian' - gaussian kernel, where 'kpar' is the gaussian sigma
    #
    #   Example:
    #       X1 = np.random.standard_normal((n, d))
    #       X2 = np.random.standard_normal((N, d))
    #       K = KernelMatrix(X1, X2, 'Linear', [])
    #       K = kernel(X1, X2, 'Gaussian', 2.0)
    #
    # See also LEARN
    
    if kernel_type == 'Linear':
        K = np.matrix(np.dot(X1,trsp(X2)))
    elif kernel_type == 'Polynomial':
        K = np.matrix(np.power((1 + np.dot(X1,trsp(X2))),KerPar))
    elif kernel_type == 'Gaussian':
        K = np.exp((-1.0)/(2.0*float(KerPar)**2.0)*SquareDist(X1,X2))
    return K