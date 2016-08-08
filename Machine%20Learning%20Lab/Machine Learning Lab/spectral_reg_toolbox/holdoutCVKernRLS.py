import numpy as np
import math

from regularizedKernLSTrain import regularizedKernLSTrain
from regularizedKernLSTest import regularizedKernLSTest
from KernelMatrix import KernelMatrix, SquareDist

def calcErr(T,Y,m):
    T=np.ravel(T)
    Y=np.ravel(Y)
    vT=(T>=m)
    vY=(Y>=m)
    compare = [vT[i]!=vY[i] for i in range(len(vT))]
    err = float(np.sum(compare))/float(np.size(Y))
    return err

def holdoutCVKernRLS(X, Y, kernel, KerPar, tmin, tmax, nt_values, space_type):
    #[l, s, Vm, Vs, Tm, Ts] = holdoutCVKernRLS(X, Y, kernel, perc, nrip, intLambda, intKerPar)
    # Xtr: the training examples
    # Ytr: the training labels
    # kernel: the kernel function (see help Gram).
    # perc: percentage of the dataset to be used for validation
    # nrip: number of repetitions of the test for each couple of parameters
    # intLambda: list of regularization parameters 
    #       for example intLambda = [5,2,1,0.7,0.5,0.3,0.2,0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001];
    # intKerPar: list of kernel parameters 
    #       for example intKerPar = [10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01];
    # 
    # Output:
    # l, s: the couple of lambda and kernel parameter that minimize the median of the
    #       validation error
    # Vm, Vs: median and variance of the validation error for each couple of parameters
    # Tm, Ts: median and variance of the error computed on the training set for each couple
    #       of parameters
    #
    # Example of code:
    # intLambda = [5,2,1,0.7,0.5,0.3,0.2,0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001];
    # intKerPar = [10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01];
    # [Xtr, Ytr] = MixGauss([[0;0],[1;1]],[0.5,0.25],100);
    # [l, s, Vm, Vs, Tm, Ts] = holdoutCVKernRLS(Xtr, Ytr,'gaussian', 0.5, 5, intLambda, intKerPar);
    # plot(intLambda, Vm, 'b');
    # hold on
    # plot(intLambda, Tm, 'r');
    nrip=51
    perc=0.5
    
    if space_type=='Linear space': 
        intLambda=np.linspace(tmin, tmax, nt_values)
    elif space_type=='Log space':
        intLambda=np.logspace(tmin, tmax, num=nt_values)
        
    intLambda=np.ndarray.tolist(intLambda)
    
    nKerPar=np.size(KerPar)
    nLambda=np.size(intLambda)
    
    
    n_KCV1=np.size(X,axis=0)
    n_KCV2=math.ceil(n_KCV1*(1-perc))
    Tm=np.zeros((nLambda,nKerPar))
    Ts=np.zeros((nLambda,nKerPar))
    Vm=np.zeros((nLambda,nKerPar))
    Vs=np.zeros((nLambda,nKerPar))
    
    ym=float((np.max(Y)+np.min(Y)))/float(2)
    
    iL=-1
    for L in intLambda:
        iL=iL+1
        iS=-1
        for S in KerPar:
            iS=iS+1
            trerr=np.zeros((nrip,1))
            vlerr=np.zeros((nrip,1))
            for rip in range(nrip):
                I=np.random.permutation(n_KCV1)
                Xtr=X[I[0:n_KCV2],:]
                Ytr=Y[I[0:n_KCV2],:]
                Xvl=X[I[n_KCV2+1:-1],:]
                Yvl=Y[I[n_KCV2+1:-1],:]
                
                w=regularizedKernLSTrain(Xtr, Ytr, kernel, S, L)
                
                y1=regularizedKernLSTest(w, Xtr, kernel, S, Xtr)
                trerr[rip]=calcErr(y1,Ytr,ym)
                
                y2=regularizedKernLSTest(w, Xtr, kernel, S, Xvl)
                vlerr[rip]=calcErr(y2,Yvl,ym)
                
                rip=rip+1
            
            Tm[iL,iS]=np.median(trerr)
            Ts[iL,iS]=np.std(trerr)
            Vm[iL,iS]=np.median(vlerr)
            Vs[iL,iS]=np.std(vlerr)
    
    row,col = np.where(Vm<=np.min(Vm))
    L=float(intLambda[int(row[0])])
    S=float(KerPar[0])
    
        
    return L, S, Vm, Vs, Tm, Ts