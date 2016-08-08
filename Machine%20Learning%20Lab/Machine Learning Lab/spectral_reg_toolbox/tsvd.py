import numpy as np

def tsvd(K, t_range, y):
    #TSVD Calculates the coefficient vector using TSVD method.
    #   [ALPHA] = TSVD(K, T_RANGE, Y) calculates the truncated singular values
    #   solution of the problem 'K*ALPHA = Y' given a kernel matrix 'K[n,n]' a 
    #   range of regularization parameters 'T_RANGE' and a label/output 
    #   vector 'Y'.
    #
    #   The function works even if 'T_RANGE' is a single value
    #
    #   Example:
    #       K = kernel('lin', [], X, X);
    #       alpha = tsvd(K, logspace(-3, 3, 7), y);
    #       alpha = tsvd(K, 0.1, y);
    #
    # See also RLS, NU, LAND, CUTOFF
    
    #if (sum(t_range > 1) + sum(t_range<0)):
        #print 'T-SVD: The range of t must be (0,s] in case of "Linear Space"; Otherwise for "Log Space" must be (-inf,log s], with s the biggest eigenvalue of the kernel matrix'
    
    n = len(y)
    
    U,S,V = np.linalg.svd(K)
    ds=np.reshape(S, (n,1))
    
    alpha = []
    for i in range(0,len(t_range)):
        t = t_range[i]
        mask = ( ds >= t*n )
        inv_ds=np.zeros_like(ds)
        for j in range(0,len(ds)):
            inv_ds[j]=(1.0/ds[j])*mask[j]
        TinvS = np.diag(np.reshape(inv_ds, (n,)))
        TK = np.linalg.multi_dot((V.T, TinvS, U.T))
        if i==0:
            alpha=np.dot(TK, y)
            alpha=np.reshape(alpha, (len(alpha),1))
        else:
            alphai=np.reshape(np.dot(TK, y),(np.size(alpha, axis=0),1))
            alpha=np.concatenate((alpha, alphai), axis=1)
        
    return alpha