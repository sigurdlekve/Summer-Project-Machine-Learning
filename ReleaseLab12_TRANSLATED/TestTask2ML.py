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

#This is a test file for testing the translated code for
# Task 2 in Release Lab 1-2. 

npoints=100
pflip=0.05
Xtr, Ytr, Xts, Yts = two_moons(npoints, pflip)

intLambda=[0.00001]
intKerPar=[10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01]
#intKerPar=[0.5]
#intLambda = [5, 2, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001]

nrip=51
perc=0.5
kernel='gaussian'

L, S, Vm, Vs, Tm, Ts = holdoutCVKernRLS(Xtr, Ytr,kernel, perc, nrip, intLambda, intKerPar)
plt.figure()
if len(intKerPar)==1:
    plt.semilogx(intLambda, Tm, label='Test error')
    plt.semilogx(intLambda, Vm, label='Validation' )
elif len(intLambda)==1:
    plt.semilogx(intKerPar, Tm.T, label='Test error')
    plt.semilogx(intKerPar, Vm.T, label='Validation')
else:
    print 'This plot is of no use, work in progress...'
plt.legend()

plt.figure()
plt.scatter(Xtr[:,0], Xtr[:,1], 25, Ytr, edgecolor='None')
c = regularizedKernLSTrain(Xtr, Ytr, kernel, S, L)
separatingFKernRLS(c, Xtr, kernel, S, Xts)

plt.figure()
plt.scatter(Xts[:,0], Xts[:,1], 25, Yts, edgecolor='None')
c = regularizedKernLSTrain(Xtr, Ytr, kernel, S, L)
separatingFKernRLS(c, Xtr, kernel, S, Xts)
plt.show()