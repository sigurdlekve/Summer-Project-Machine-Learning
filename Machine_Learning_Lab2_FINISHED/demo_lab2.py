import numpy as np
import scipy.io as sio

from spectral_reg_toolbox import kcv
from spectral_reg_toolbox import autosigma

from save_challenge_2 import save_challenge_2

def demo_lab2():

    # this part will load the dataset
    zero_train=sio.loadmat('zero_train.mat')
    three_train=sio.loadmat('three_train.mat')
    eight_train=sio.loadmat('eight_train.mat')
    
    TRAIN_ZEROS=zero_train['zero_train']
    TRAIN_THREES=three_train['three_train']
    TRAIN_EIGHTS=eight_train['eight_train']
    X=np.concatenate((TRAIN_ZEROS, TRAIN_THREES, TRAIN_EIGHTS), axis=0)
    
    #Trivial Part
    YOURNAME = ... # eg. 'john_smith' pay attention to the underscore
    
    #Challenging Part
    FILT= ...                           # eg. 'Truncated SVD' see 'kcv'
    N_SPLIT = ...                                  # eg. 5     see 'kcv'
    SPLIT_TYPE = ...                        # eg. 'Sequential' see 'kcv'
    KERNEL = ...                             # eg. 'Linear' see 'KernelMatrix'
    KERNEL_PARAMETER = ...            #fix it manually or by autosigma for example with autosigma(X,5). see 'KernelMatrix' 'kcv' and 'autosigma'
    TRANGE = ...                      # eg. np.logspace(-3, 3, 7) or np.linspace(0.1, 10, 10)
    
    
    #Perform the 1 vs. all procedure
    t=np.zeros((3,1))
    errval=np.zeros((3,1))
    for i in range(0,3):
        Y=np.ones((900, 1))*-1
        Y[(300*i):(300*i+300),0]=Y[(300*i):(300*i+300),0]*-1
        #Perform the k-fold cross validation
        t_kcv_idx, avg_err_kcv = kcv(KERNEL, KERNEL_PARAMETER, FILT, TRANGE, X, Y, N_SPLIT, 'Classification', SPLIT_TYPE);
        if FILT=='Landweber' or FILT=='NU-method':
            t[i,0]=t_kcv_idx
        else:
            t[i,0]=TRANGE[t_kcv_idx]
        
        errval[i,0]=avg_err_kcv[0][t_kcv_idx]
       
    save_challenge_2(YOURNAME, t, KERNEL, KERNEL_PARAMETER, errval, FILT)
    
    return
