'''
Created on 29. jun. 2016

@author: Sigurd Lekve
'''
import numpy as np
from numpy import transpose as trsp
import matplotlib.pyplot as plt
import scipy.io as sio
import time as time
from KernelMatrix import KernelMatrix
from MixGauss import MixGauss
from regularizedKernLSTest import regularizedKernLSTest
from regularizedKernLSTrain import regularizedKernLSTrain
from separatingFKernRLS import separatingFKernRLS
from SquareDist import SquareDist
from flipLabels import flipLabels
from two_moons import two_moons
from holdoutCVKernRLS import holdoutCVKernRLS, calcErr

start=time.time()

npoints=50
pflip=0.05
Xtr, Ytr, Xts, Yts = two_moons(npoints, pflip)

#intKerPar=[0.5]
intKerPar=[10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01]
#intLambda = [5, 2, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001]
intLambda=[0.00001]
nrip=51
perc=0.5
kernel='gaussian'
l=[0.000001, 0.001, 1, 10]
sigma=[0.1, 1, 5, 10]

L, S, Vm, Vs, Tm, Ts = holdoutCVKernRLS(Xtr, Ytr,kernel, perc, nrip, intLambda, intKerPar)
plt.figure()
if len(intKerPar)==1:
    plt.semilogx(intLambda, Tm)
    plt.semilogx(intLambda, Vm)
elif len(intLambda)==1:
    plt.semilogx(intKerPar, Tm.T)
    plt.semilogx(intKerPar, Vm.T)
else:
    print 'Dritt plot'

plt.figure()
plt.scatter(Xtr[:,0], Xtr[:,1], 25, Ytr, edgecolor='None')
c = regularizedKernLSTrain(Xtr, Ytr, kernel, S, L)
separatingFKernRLS(c, Xtr, kernel, S, Xts)

plt.figure()
plt.scatter(Xts[:,0], Xts[:,1], 25, Yts, edgecolor='None')
c = regularizedKernLSTrain(Xtr, Ytr, kernel, S, L)
separatingFKernRLS(c, Xtr, kernel, S, Xts)
#plt.show()

end=time.time()
print (end-start)

#plt.figure()
#plt.scatter(Xtr[:,0], Xtr[:,1], 25, Ytr, edgecolor='None')

#c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[1])
#separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
# c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[1])
# separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
# c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[2])
# separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
# c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[3])
# separatingFKernRLS(c, Xtr, kernel, sigma, Xts)


#plt.figure()
#plt.scatter(Xts[:,0], Xts[:,1], 25, Yts, edgecolor='None')

#c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[1])
#separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
# c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[1])
# separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
# c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[2])
# separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
# c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[3])
# separatingFKernRLS(c, Xtr, kernel, sigma, Xts)