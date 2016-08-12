import numpy as np
import scipy.io as sio

from spectral_reg_toolbox import kcv
from spectral_reg_toolbox import autosigma

def demo_lab1():

    # this part will load the dataset
    one_train=sio.loadmat('one_train.mat')
    seven_train=sio.loadmat('seven_train.mat')
    
    TRAIN_ONES=one_train['one_train']
    TRAIN_SEVENS=seven_train['seven_train']
    TRAIN=np.concatenate((TRAIN_ONES, TRAIN_SEVENS), axis=0)
    
    LABEL_ONES=np.zeros((300,1))
    LABEL_SEVENS=np.zeros((300,1))*-1
    LABEL=np.concatenate((LABEL_ONES, LABEL_SEVENS), axis=0)
 
    #Trivial Part
    YOURNAME = 'sigurd_lekve' # eg. 'john_smith' pay attention to the underscore
          
    #Challenging Part
    N_SPLIT = 5                                      # eg. 5     see 'kcv'
    SPLIT_TYPE = 'Sequential'                        # eg. 'Sequential' see 'kcv'
    KERNEL = 'Polynomial'                              # eg. 'Linear' see 'KernelMatrix'
    KERNEL_PARAMETER = 10 #autosigma(TRAIN,5)            #fix it manually or by autosigma for example with autosigma(TRAIN,5). see 'KernelMatrix' 'kcv' and 'autosigma'
    TRANGE =  np.linspace(0.1, 1, 10)               # eg. logspace(-3, 3, 7);
    

    t_kcv_idx, avg_err_kcv = kcv(KERNEL, KERNEL_PARAMETER, 'Reg. Least Squared', TRANGE, TRAIN, LABEL, N_SPLIT, 'Classification', SPLIT_TYPE)
    print TRANGE
    print t_kcv_idx
    print TRANGE[t_kcv_idx]
    print avg_err_kcv
    print avg_err_kcv[0][t_kcv_idx]
    #save_challenge_1(YOURNAME, TRANGE[t_kcv_idx], KERNEL, KERNEL_PARAMETER, avg_err_kcv[0][t_kcv_idx])
    return

demo_lab1()
