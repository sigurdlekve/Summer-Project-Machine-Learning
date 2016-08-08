import numpy as np
import math

def nu(K, t_max, y, all_path=False):
    #NU Calculates the coefficient vector using NU method.
    #   [ALPHA] = NU(K, T_MAX, Y) calculates the solution of the problem 
    #   'K*ALPHA = Y' using NU method given a kernel matrix 
    #   'K[n,n]', the maximum number of the iterations 'T_MAX' and a 
    #   label/output vector 'Y'.
    #
    #   [ALPHA] = NU(K, T_MAX, Y, ALL_PATH) returns only the last 
    #   solution calculated using 'T_MAX' as regularization parameter if
    #   'ALL_PATH' is false(DEFAULT). Otherwise return all the regularization 
    #   path.
    #
    #   Example:
    #       K = kernel('lin', [], X, X);
    #       alpha = nu(K, 10, y);
    #       alpha = nu(K, 10, y, true);
    #
    # See also TSVD, CUTOFF, RLS, LAND
    
    t_max = math.floor(t_max[0])
    if (t_max < 2):
        print 't_max must be an int greater than 1','Tips and tricks'
        
    n=np.size(y, axis=0)
    alpha = []
    alpha=np.zeros((n,2))
    
    #alpha{1} = zeros(n,1);
    #alpha{2} = zeros(n,1);
    #nu = 1;
    
    nu=1.0
    j=3.0
    t_max=int(t_max)
    for i in range(2,t_max):
        u = ( (j-1.0) * (2.0*j-3.0) * (2.0*j+2.0*nu-1.0) ) / ( (j+2.0*nu-1.0) * (2.0*j+4.0*nu-1.0) * (2.0*j+2.0*nu-3.0) )
        w = 4.0 * ( ((2.0*j+2.0*nu-1.0)*(j+nu-1.0) ) / ( (j+2.0*nu-1.0)*(2.0*j+4.0*nu-1.0)) )
        
        alpha1=np.reshape(alpha[:,i-2],(np.size(alpha[:,i-2], axis=0),1) )
        alpha2=np.reshape(alpha[:,i-1],(np.size(alpha[:,i-1], axis=0),1) )
        alpha3 = alpha2 + u*( alpha2 - alpha1 ) + (float(w)/float(n)) * (y - np.dot(K, alpha2))
        alpha=np.concatenate((alpha,alpha3), axis=1)
        j=j+1
        
    if all_path==False:
        alpha = alpha[:, t_max-1]
        
    return alpha