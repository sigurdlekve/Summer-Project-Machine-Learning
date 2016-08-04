'''
Created on 22. jun. 2016

@author: Sigurd Lekve
'''
import numpy as np
import matplotlib.pyplot as plt

from dataset_scripts import gaussian
from spectral_reg_toolbox import KernelMatrix, SquareDist
from spectral_reg_toolbox import tsvd
from spectral_reg_toolbox import learn
from spectral_reg_toolbox import learn_error
from spectral_reg_toolbox import splitting
from spectral_reg_toolbox import kcv

import scipy.io as sio

#B=np.zeros((3,3))
#A=np.random.standard_normal((10,3))
#C=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
#Ct=np.transpose(C)
#a=[[0],[0]]
#b=[[1],[1]]
#ab=[a,b]
#c=[[0.5],[0.25]]
#n=1000
#MixGauss(ab,c,n)
#BnCt=np.concatenate(B,Ct)

#t=np.size(B, axis=1)

#kernel='linear'

#print(np.array_equal(kernel, 'linear'))

#A=np.ones((4,2))*3
#B=np.ones((2,4))*4
#print A
#print B
#AB=np.dot(A,B)
#print AB
#AdivB=A/B

#x1 = np.arange(9.0).reshape((3, 3))
#x2 = np.arange(9.0).reshape((3, 3))
#x3=np.multiply(x1, x2)

#a=[[0],[0]]
#b=[[1],[1]]
#ab=[a,b]
#c=[[0.5],[0.3]]
#n=5
#X,Y=MixGauss(ab,c,n)

#a=np.arange(1,11)
#reshape(a)
#x1=[[0],[0]]

#print x1
#x2=np.zeros((2,1))
#print x2

# origin = 'lower'
# #origin = 'upper'
# 
# delta = 0.025
# 
# x = y = np.arange(-3.0, 3.01, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = plt.mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
# Z2 = plt.mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# Z = 10 * (Z1 - Z2)
# 
# nr, nc = Z.shape
# 
# # put NaNs in one corner:
# Z[-nr//6:, -nc//6:] = np.nan
# # contourf will convert these to masked
# 
# 
# Z = np.ma.array(Z)
# # mask another corner:
# Z[:nr//6, :nc//6] = np.ma.masked
# 
# # mask a circle in the middle:
# interior = np.sqrt((X**2) + (Y**2)) < 0.5
# Z[interior] = np.ma.masked

# npoints=100
# print(np.min([npoints, 100]))

# nrip=51
# print np.arange(1,nrip+1)
# a=np.random.permutation(100).reshape((10,10))
# print a
# i, j = np.where(a<=np.min(a))
# print int(i)
# #i,j=np.unravel_index(a.argmin(),a.shape)
# print i,j
#a=np.array([0, 0])
#b=np.array([1, 1])
#a=[[0],[0]]
#b=[[1],[1]]
#ab=[a,b]
#print a, b
a=np.array([0.1, 0.5, 0.7])
#b=len(a)
print a
#print b
#pung=np.array(np.where(Y>=0))
#pung=np.array([pung[0][:]])
#print pung
#print np.size(pung, axis=1)
# test_dataset=sio.loadmat('C:\Users\Sigurd Lekve\Documents\MATLAB\Simula\Lab 1\spectral_reg_toolbox\\testdata_python.mat'
# 
# X=test_dataset['X']
# Y=test_dataset['Y']
# print 'X', X
# print 'Y', Y
#K = KernelMatrix(X, X, 'Linear', [])
#alpha = tsvd(K, [0.1, 0.5, 0.7], Y)
#print alpha
#X, Y=gaussian([3,3], 0)
#alpha, err = learn('Linear',[], 'tsvd', a, X, Y, 'class')
#print alpha, err
#sets=splitting(Y, 3, type='seq')
#print sets
# trange=[0.0, 0.25, 0.5, 0.75, 1]
# test=np.linspace(0.1,10,10)
# print test

#t_kcv_idx, avg_err_kcv=kcv('', 1, 'tsvd', trange, X, Y, 3, 'Classification', 'Sequential')


# t1=np. array([[2.0, 5.0], [5.0, 2.0]])
# print type(t1),t1, np.shape(t1)
# t2=np.array([[3],[7]])
# print type(t2), t2, np.shape(t2)
# print np.dot(t1, t2)
# 
# print (7.25110708 * -13.6921881) + (-17.46558814 * 5.08743318) + (19.09320513 * 8.03566375) + (17.32096178 * 2.00404334)


test=np.linspace(0.1,10,10)
print test
test1=np.reshape(test, (1, len(test)))
print test1

