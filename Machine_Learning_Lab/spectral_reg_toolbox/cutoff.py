import numpy as np

def cutoff(K, t_range, y):
    #CUTOFF Calculates the coefficient vector using cutoff method.
    #   ALPHA = CUTOFF(K, T_RANGE, Y) calculates the spectral cut-off 
    #   solution of the problem 'K*ALPHA = Y' given a kernel matrix 'K[n,n]' a 
    #   range of regularization parameters 'T_RANGE' and a label/output 
    #   vector 'Y'.
    #
    #   The function works even if 'T_RANGE' is a single value
    #
    #   Example:
    #       K = kernel('lin', [], X, X)
    #       alpha = cutoff(K, 0.1, y)
    #
    # See also RLS, NU, TSVD, LAND
    
    n = np.size(y, axis=0)
    alpha = []
    
    U,S,V = np.linalg.svd(K)
    ds=np.reshape(S, (n,1))
    for i in range(0, len(t_range)):
        t = t_range[i]
        mask = ( ds > t*n )
        index = np.sum(mask)
        inv_ds1=np.zeros_like(ds[0:index])
        for j in range(0, len(inv_ds1)):
            inv_ds1[j]=1.0/ds[j]
        
        inv_ds2=(1.0/(t*n))*np.ones((n-index, 1))
        inv_ds=np.concatenate((inv_ds1,inv_ds2))
        TinvS = np.diag(np.reshape(inv_ds,(n,)))
        #TK = np.linalg.multi_dot((V.T, TinvS, U.T))
        TK=np.dot(np.dot(V.T,TinvS), U.T)
        if i==0:
            alpha=np.dot(TK, y)
            alpha=np.reshape(alpha, (len(alpha),1))
        else:
            alphai=np.reshape(np.dot(TK, y),(np.size(alpha, axis=0),1))
            alpha=np.concatenate((alpha, alphai), axis=1)
            
    return alpha