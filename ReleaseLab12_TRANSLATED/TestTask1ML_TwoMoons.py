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

#This is a test file for testing the translated code for
# Task 1 with the dataset "Two Moons" in Release Lab 1-2. 

npoints=50;
pflip=0.05;
Xtr, Ytr, Xts, Yts = two_moons(npoints, pflip);
kernel='gaussian'
l=[0.000001, 0.001, 1, 10]
sigma=[0.1, 1, 5, 10]
sigma=sigma[1]

plt.figure(1)
plt.scatter(Xtr[:,0], Xtr[:,1], 25, Ytr, edgecolor='None')

c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[0])
separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[1])
separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[2])
separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[3])
separatingFKernRLS(c, Xtr, kernel, sigma, Xts)


plt.figure(2)
plt.scatter(Xts[:,0], Xts[:,1], 25, Yts, edgecolor='None')

c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[0])
separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[1])
separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[2])
separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, l[3])
separatingFKernRLS(c, Xtr, kernel, sigma, Xts)

plt.show()