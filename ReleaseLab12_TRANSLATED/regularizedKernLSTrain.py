'''
Created on 22. jun. 2016

@author: Sigurd Lekve
'''
import numpy as np
from KernelMatrix import KernelMatrix
#l=lambda

def regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l):
    n = np.size(Xtr,axis=0)
    K = KernelMatrix(Xtr, Xtr, kernel, sigma)
    c =np.linalg.solve((K + l * n * np.eye(n)), Ytr)
    return c

#Test
