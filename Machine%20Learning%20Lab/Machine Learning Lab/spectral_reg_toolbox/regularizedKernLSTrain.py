import numpy as np
from KernelMatrix import KernelMatrix, SquareDist

def regularizedKernLSTrain(Xtr, Ytr, kernel_type, sigma, l):
    n = np.size(Xtr,axis=0)
    K = KernelMatrix(Xtr, Xtr, kernel_type, sigma)
    c =np.linalg.solve((K + (l * n * np.eye(n))), Ytr)
    return c