import numpy as np
import scipy.io as sio

from spectral_reg_toolbox import kcv
from spectral_reg_toolbox import autosigma

from save_challenge_1 import save_challenge_1

def demo_lab1():

    # this part will load the dataset
    one_train=sio.loadmat('one_train.mat')
    seven_train=sio.loadmat('seven_train.mat')
    
    TRAIN_ONES=one_train['one_train']
    TRAIN_SEVENS=seven_train['seven_train']
    TRAIN=np.concatenate((TRAIN_ONES, TRAIN_SEVENS), axis=0)
    
    LABEL_ONES=np.ones((300,1))
    LABEL_SEVENS=np.ones((300,1))*-1
    LABEL=np.concatenate((LABEL_ONES, LABEL_SEVENS), axis=0)
 
    #Trivial Part
    YOURNAME = ...          # eg. 'john_smith' pay attention to the underscore
          
    #Challenging Part
    N_SPLIT =  ...                                 # eg. 5     see 'kcv'
    SPLIT_TYPE = ...                        # eg. 'Sequential' see 'kcv'
    KERNEL = ...                              # eg. 'Linear' see 'KernelMatrix'
    KERNEL_PARAMETER = ...           #fix it manually or by autosigma for example with autosigma(TRAIN,5). see 'KernelMatrix' 'kcv' and 'autosigma'
    TRANGE =  ...                      # eg. np.logspace(-3, 3, 7) or np.linspace(0.1, 10, 10)

    t_kcv_idx, avg_err_kcv = kcv(KERNEL, KERNEL_PARAMETER, 'Reg. Least Squared', TRANGE, TRAIN, LABEL, N_SPLIT, 'Classification', SPLIT_TYPE)
    save_challenge_1(YOURNAME, TRANGE[t_kcv_idx], KERNEL, KERNEL_PARAMETER, avg_err_kcv[0][t_kcv_idx])

    return
