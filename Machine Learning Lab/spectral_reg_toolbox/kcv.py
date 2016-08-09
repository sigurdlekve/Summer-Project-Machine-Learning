import numpy as np
import math
import scipy.sparse.linalg

from learn import learn
from learn_error import learn_error
from KernelMatrix import KernelMatrix, SquareDist
from tsvd import tsvd
from rls import rls
from cutoff import cutoff
from land import land
from nu import nu
from splitting import splitting

def kcv(knl, kpar, filt, t_range, X, y, k, task, split_type):
    #KCV Perform K-Fold Cross Validation.
    #   [T_KCV_IDX, ERR_KCV] = KCV(KNL, KPAR, FILT, T_RANGE, X, Y, K, TASK, SPLIT_TYPE) 
    #   performs k-fold cross validation to calculate the index of the 
    #   regularization parameter 'T_KCV_IDX' within a range 'T_RANGE' 
    #   which minimizes the average cross validation error 'AVG_ERR_KCV' given 
    #   a kernel type 'KNL' and (if needed) a kernel parameter 'KPAR', a filter
    #   type 'FILT' and a dataset composed by the input matrix 'X[n,d]' and the output vector 
    #   'Y[n,1]'.
    #
    #   The allowed values for 'KNL' and 'KPAR' are described in the
    #   documentation given with the 'KERNEL' function. Moreover, it is possible to
    #   specify a custom kernel with 'KNL='cust'' and 'KPAR[n,n]' matrix.
    #
    #   The allowed values for 'FILT' are:
    #       'rls'   - regularized least squares
    #       'land'  - iterative Landweber
    #       'tsvd'  - truncated SVD
    #       'nu'    - nu-method
    #       'cutoff'- spectral cut-off
    #   
    #   The parameter 'T_RANGE' may be a range of values or a single value.
    #   In case 'FILT' equals 'land' or 'nu', 'T_RANGE' *MUST BE* a single
    #   integer value, because its value is interpreted as 'T_MAX' (see also 
    #   LAND and NU documentation). Note that in case of 'land' the algorithm
    #   step size 'tau' will be automatically calculated (and printed).
    #
    #   According to the parameter 'TASK':
    #       'class' - classification
    #       'regr'  - regression
    #   the function minimizes the classification or regression error.
    #
    #   The last parameter 'SPLIT_TYPE' must be:
    #       'seq' - sequential split (as default)
    #       'rand' - random split
    #   as indicated in the 'SPLITTING' function.
    #
    #   Example:
    #       [t_kcv_idx, avg_err_kcv] = kcv('lin', [], 'rls', logspace(-3, 3, 7), X, y, 5, 'class', 'seq')
    #       [t_kcv_idx, avg_err_kcv] = kcv('gauss', 2.0, 'land', 100, X, y, 5, 'regr', 'rand')
    #
    # See also LEARN, KERNEL, SPLITTING, LEARN_ERROR
    
    k=math.ceil(k)
    if (k <= 1):
        print 'The number of splits in KCV must be at least 2'
    
    ## Split of training set:
    k=int(k)
    sets = splitting(y, k, split_type) 
    
    ## Starting Cross Validation
    err_kcv=[]
    for i in range(0,k):
        err_kcv.append([])  #one series of errors for each split
    for split in range(0,k):
        print 'Split number', split
    
        test_idxs = sets[split]
        train_idxs = np.setdiff1d(np.arange(0,np.size(y,axis=0)), test_idxs)
        
        X_train = X[train_idxs, :]
        y_train = y[train_idxs, 0]
        y_train = np.reshape(y_train, (len(y_train),1))
    
        X_test = X[test_idxs, :]
        y_test = y[test_idxs, 0]
        y_test = np.reshape(y_test, (len(y_test),1))

        ## Learning
        alpha, err =  learn(knl, kpar, filt, t_range, X_train, y_train, task)
        print err
        
        ## Test error estimation
        # Error estimation over the test set, using the parameters given by the previous task.
        K_test = KernelMatrix(X_test, X_train, knl, kpar)
        init_err_kcv=np.zeros((1, np.size(alpha, axis=1)))
        init_err_kcv=np.reshape(init_err_kcv, np.size(alpha, axis=1))
        init_err_kcv=list(init_err_kcv)
        
        # On each split we estimate the error with each t value in the range
        err_kcv[split]=init_err_kcv
        for t in range(0, np.size(alpha, axis=1)):
            y_learnt = np.dot(K_test, alpha[:,t])
            err_kcv[split][t] =learn_error(y_learnt, y_test, task)
    print err_kcv
               
    ## Average the error over different splits
    err_kcv=np.reshape(err_kcv, (np.size(err_kcv, axis=0), len(err_kcv[0])))
    avg_err_kcv =[]
    for l in range(0, np.size(err_kcv, axis=1)):
        avg_err_kcv.append(np.median(err_kcv[:,l]))
    
    ## Calculate minimum error w.r.t. the regularization parameter
    
    #min_err = inf
    t_kcv_idx = -1;
    avg_err_kcv=np.reshape(avg_err_kcv, (1, len(avg_err_kcv)))
    ny=np.size(avg_err_kcv, axis=0)
    nx=np.size(avg_err_kcv, axis=1)
    
    for i in range(0, (nx*ny)):
        if i==0:
           min_err = avg_err_kcv[0,i]
           t_kcv_idx = i 
        elif avg_err_kcv[0,i] <= min_err:
            min_err = avg_err_kcv[0,i]
            t_kcv_idx = i
    
    #np.size(alpha[t_kcv_idx], axis=0)
    #np.size(alpha[t_kcv_idx], axis=1)
    #alpha[t_kcv_idx]
    return t_kcv_idx, avg_err_kcv