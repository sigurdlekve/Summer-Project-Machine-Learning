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
    if kernel_type == 'Linear':
        K = np.dot(X1,trsp(X2))
    elif kernel_type == 'Polynomial':
        K = np.power((1 + np.dot(X1,trsp(X2))),KerPar)
    elif kernel_type == 'Gaussian':
        K = np.exp((-1.0)/(2.0*float(KerPar)**2.0)*SquareDist(X1,X2))
    return K