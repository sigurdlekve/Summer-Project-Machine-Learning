import numpy as np
import math

def land(K, t_max, y, tau, all_path=False):
    #LAND Calculates the coefficient vector using Landweber method.
    #   ALPHA = LAND(K, T_MAX, Y, TAU) calculates the regularized least 
    #   squares  solution of the problem 'K*ALPHA = Y' given a kernel matrix 
    #   'K[n,n]' a maximum regularization parameter 'T_MAX', a
    #   label/output vector 'Y' and a step size 'TAU'.
    #
    #   ALPHA = LAND(K, T_MAX, Y, TAU, ALL_PATH) returns only the last 
    #   solution calculated using 'T_MAX' as regularization parameter if
    #   'ALL_PATH' is false. Otherwise return all the regularization path.
    #
    #   Example:
    #       K = KernelMatrix(X, X, 'Linear', [])
    #       alpha = land(K, 10, y, 2)
    #       alpha = land(K, 10, y, 2, all_path=True)
    #
    # See also NU, TSVD, CUTOFF, RLS

    t_max = math.floor(t_max[0]);
    if (t_max < 1):
        print 't_max must be an int greater equal than 1'
        
    n = np.size(y, axis=0)
    alpha=[]
    
    alpha=np.zeros((n,1))
    t_max=int(t_max)
    for i in range(1, t_max):
        alphai=np.reshape(alpha[:,i-1], (len(alpha[:,i-1]),1)) + (float(tau)/float(n)) * (y - np.dot(K,np.reshape(alpha[:,i-1], (len(alpha[:,i-1]),1))))
        alpha=np.concatenate((alpha,alphai), axis=1)
    
    if all_path==False:
        alpha = alpha[:,t_max-1]    
    return alpha