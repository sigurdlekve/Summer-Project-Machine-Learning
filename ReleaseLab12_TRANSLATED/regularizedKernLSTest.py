'''
Created on 22. jun. 2016

@author: Sigurd Lekve
'''
import numpy as np
from KernelMatrix import KernelMatrix

def regularizedKernLSTest(c, Xtr, kernel, sigma, Xts):
    Ktest = KernelMatrix(Xts, Xtr, kernel, sigma)
    y = np.dot(Ktest,c)
    return y
