import numpy as np
from numpy import transpose as trsp
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from regularizedKernLSTest import regularizedKernLSTest
from KernelMatrix import KernelMatrix, SquareDist

def separatingFKernRLS(c, Xtr, kernel, sigma, Xts):
    # function separatingFKernRLS(w, Xtr, kernel, sigma, Xts)
    # the function classifies points evenly sampled in a visualization area,
    # according to the classifier Regularized Least Squares
    #
    # c - coefficents of the function
    # Xtr - training examples
    # kernel, sigma - parameters used in learning the function
    # Xts - test examples on which to plot the separating function
    #
    # lambda = 0.01;
    # kernel = 'gaussian';
    # sigma = 1;
    # [Xtr, Ytr] = MixGauss([[0;0],[1;1]],[0.5,0.25],1000);
    # [Xts, Yts] = MixGauss([[0;0],[1;1]],[0.5,0.25],1000);
    # Ytr(Ytr==2) = -1;
    # Yts(Yts==2) = -1;
    # c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, lambda);
    # separatingFKernRLS(c, Xtr, kernel, sigma, Xts);
    step = 0.05
    x=np.arange(min(Xts[:,0]),max(Xts[:,0]),step)
    y=np.arange(min(Xts[:,1]),max(Xts[:,1]),step)
    
    X,Y=np.meshgrid(x,y)
    nX=np.reshape(trsp(X),(np.size(X),1))
    nY=np.reshape(trsp(Y),(np.size(Y),1))
    
    XGrid=np.column_stack((nX,nY))
    YGrid = regularizedKernLSTest(c, Xtr, kernel, sigma, XGrid)
    
    z=np.array(np.reshape(YGrid,(np.size(X,axis=1),np.size(X,axis=0))).T)
    z=np.array(z)
    
    return X, Y, z