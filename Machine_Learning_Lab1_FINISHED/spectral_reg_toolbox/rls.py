import numpy as np

def rls(K, t_range, y):
    #   RLS Calculates the coefficient vector using Tikhonov method.
    #   ALPHA = RLS(K, lambdas, Y) calculates the least squares solution
    #   of the problem 'K*ALPHA = Y' given a kernel matrix 'K[n,n]' a 
    #   range of regularization parameters 'lambdas' and a label/output 
    #   vector 'Y'.
    #
    #   The function works even if 'T_RANGE' is a single value
    #
    #   Example:
    #       K = KernelMatrix(X, X, 'Linear', [])
    #       alpha = rls(K, np.linspace(1,10,20), y)
    #       alpha = rls(K, 0.1, y)
    #
    # See also NU, TSVD, LAND, CUTOFF
    
    n = np.size(y, axis=0)
    alpha = []
    
    U,S,V = np.linalg.svd(K)
    ds=np.reshape(S, (n,1))
    
    for i in range(0,len(t_range)):
        t = t_range[i]
        dsi=ds + (t*n)
        TikS_temp=np.zeros_like(ds)
        for j in range(0, len(ds)):
            TikS_temp[j]=1.0/dsi[j]
        
        TikS_temp=np.reshape(TikS_temp, (n,))
        TikS=np.diag(TikS_temp)
        #TikK=np.linalg.multi_dot((V.T, TikS, U.T))
        TikK=np.dot(np.dot(V.T,TikS), U.T)
        if i==0:
            alpha=np.dot(TikK, y)
            alpha=np.reshape(alpha, (len(alpha),1))
        else:
            alphai=np.reshape(np.dot(TikK, y),(np.size(alpha, axis=0),1))
            alpha=np.concatenate((alpha, alphai), axis=1)
    return alpha