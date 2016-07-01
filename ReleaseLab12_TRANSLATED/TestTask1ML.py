'''
Created on 23. jun. 2016

@author: Sigurd Lekve
'''
import numpy as np
from numpy import transpose as trsp
import matplotlib.pyplot as plt
import time as time
from KernelMatrix import KernelMatrix
from MixGauss import MixGauss
from regularizedKernLSTest import regularizedKernLSTest
from regularizedKernLSTrain import regularizedKernLSTrain
from separatingFKernRLS import separatingFKernRLS
from SquareDist import SquareDist
from flipLabels import flipLabels


#Test of tasl 1: Kernel Regularized Least Squares
start=time.time()

Xtr, Ytr = MixGauss([[[0],[0]],[[1],[1]]],[[0.5],[0.3]],100)
Xts, Yts = MixGauss([[[0],[0]],[[1],[1]]],[[0.5],[0.3]],100)
Ytr[Ytr==2]=-1
Ytr[Ytr==2]=-1

plt.figure(1)
plt.scatter(Xtr[:,0],Xtr[:,1],50,Ytr,edgecolor='None')

l=[0.00001, 0.01, 1, 10]
kernel='gaussian'
sigma=1

c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[1])
separatingFKernRLS(c, Xtr, kernel, sigma, Xts)

# c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[1])
# separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
# c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[2])
# separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
# c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[3])
# separatingFKernRLS(c, Xtr, kernel, sigma, Xts)

p=0.1;
Ytrn=flipLabels(Ytr,p)

plt.figure(2)
plt.scatter(Xtr[:,0], Xtr[:,1], 50, Ytrn, edgecolor='None')

c = regularizedKernLSTrain(Xtr, Ytrn, kernel, sigma, l[1])
separatingFKernRLS(c, Xtr, kernel, sigma, Xts)

# c = regularizedKernLSTrain(Xtr, Ytrn, kernel, sigma, l[1])
# separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
# c = regularizedKernLSTrain(Xtr, Ytrn, kernel, sigma, l[2])
# separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
# c = regularizedKernLSTrain(Xtr, Ytrn, kernel, sigma, l[3])
# separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
#plt.show()

end=time.time()
print'ttot',(end-start)