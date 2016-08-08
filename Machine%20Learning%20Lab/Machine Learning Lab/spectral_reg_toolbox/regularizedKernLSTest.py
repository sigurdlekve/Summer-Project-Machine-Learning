import numpy as np
from KernelMatrix import KernelMatrix, SquareDist

def regularizedKernLSTest(c, Xtr, kernel, sigma, Xts):
    Ktest = KernelMatrix(Xts, Xtr, kernel, sigma)
    y = np.dot(Ktest,c)
    return y