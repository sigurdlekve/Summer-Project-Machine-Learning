import numpy as np

def EuclideanDistance(a,b,df=0):
    # EUCLIDEAN - computes Euclidean distance matrix
    #
    # E = euclidean(A,B)
    #
    #    A - (DxM) matrix 
    #    B - (DxN) matrix
    #    df = 1, force diagonals to be zero; 0 (default), do not force
    # 
    # Returns:
    #    E - (MxN) Euclidean distances between vectors in A and B
    #
    #
    # Description : 
    #    This fully vectorized (VERY FAST!) m-file computes the 
    #    Euclidean distance between two vectors by:
    #
    #                 ||A-B|| = sqrt ( ||A||^2 + ||B||^2 - 2*A.B )
    #
    # Example : 
    #    A = rand(400,100); B = rand(400,200);
    #    d = distance(A,B);
    
    # Author   : Roland Bunschoten
    #            University of Amsterdam
    #            Intelligent Autonomous Systems (IAS) group
    #            Kruislaan 403  1098 SJ Amsterdam
    #            tel.(+31)20-5257524
    #            bunschot@wins.uva.nl
    # Last Rev : Wed Oct 20 08:58:08 MET DST 1999
    # Tested   : PC Matlab v5.2 and Solaris Matlab v5.3
    
    # Copyright notice: You are free to modify, extend and distribute 
    #    this code granted that the author of the original code is 
    #    mentioned as the original author of the code.
    
    # Fixed by JBT (3/18/00) to work for 1-dimensional vectors
    # and to warn for imaginary numbers.  Also ensures that 
    # output is all real, and allows the option of forcing diagonals to
    # be zero.  
    a=a.T
    b=b.T
    
    if (np.size(a,axis=0) != np.size(b,axis=0)):
       print 'A and B should be of same dimensionality'
    
    if np.ndarray.tolist(np.isreal(a)).count(0)>0 or np.ndarray.tolist(np.isreal(b)).count(0)>0:
       print 'Warning: running L2Distance.m with imaginary numbers.  Results may be off.' 
    
    if (np.size(a, axis=0) == 1):
      a = np.concatenate((a, np.zeros((1,np.size(a,axis=1))))) 
      b = np.concatenate((b, np.zeros((1,np.size(b,axis=1))))) 
      
    aa=np.sum(np.matrix(np.multiply(a,a)), axis=0)
    bb=np.sum(np.matrix(np.multiply(b,b)), axis=0)
    ab=np.dot(a.T,b)
    d1=np.tile(aa.T,(1, np.size(bb,axis=0)))
    d2=np.tile(bb,(np.size(aa,axis=1), 1))
    d3=2*ab
    d = np.sqrt(d1+d2-d3)
                
    # make sure result is all real
    d = np.real(d) 
    
    # force 0 on the diagonal? 
    if (df==1):
      np.fill_diagonal(d, 0)
      
    return d