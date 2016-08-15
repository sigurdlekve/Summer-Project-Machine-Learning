import numpy as np

def splitting(y, k, split_type='Sequential'):
    # SPLITTING Calculate cross validation splits.
    #   SETS = SPLITTING(Y, K) splits a dataset to do K-Fold Cross validation 
    #   given a labels vector 'Y', the number of splits 'K'.
    #   Returns a list of arrays array of 'K' subsets of the indexes 1:n, with 
    #   n=length(Y). The elements 1:n are split so that in each 
    #   subset the ratio between indexes corresponding to positive elements 
    #   of array 'Y' and indexes corresponding to negative elements of 'Y' is 
    #   the about same as in 1:n. 
    #   As default, the subsets are obtained  by sequentially distributing the 
    #   elements of 1:n.
    #
    #   SETS = SPLITTING(Y, K, TYPE) allows to specify the 'TYPE' of the
    #   splitting of the chosen from
    #       'Sequential' - sequential split (as default)
    #       'Random' - random split
    #
    #    Example:
    #       sets = splitting(y, k)
    #       sets = splitting(y, k, 'Random')
    #
    # See also KCV

    if k <= 0:
        print 'Parameter k MUST be an integer greater than 0'

    if (split_type=='Sequential' or split_type=='Random')==False:
        print 'type must be Sequential or Random'
 
    n = np.size(y, axis=0)
    if k==n:  #Leave-One-Out
        sets = np.zeros((1,n))
        for i in range(0,n):
            sets[0][i] = i
    else:
        c1=np.array(np.where(y>=0))
        c1=np.array([c1[0][:]])
        #c1=np.add(c1,np.ones_like(c1))
        c2=np.array(np.where(y<0))
        c2=np.array([c2[0][:]])
        #c2=np.add(c2,np.ones_like(c2))
        l1 = np.size(c1, axis=1)
        l2 = np.size(c2, axis=1)
        
        if split_type=='Sequential':
            perm1 = np.reshape(np.arange(0,l1),(1,l1))
            perm2 = np.reshape(np.arange(0,l2),(1,l2))
        elif split_type=='Random':
            perm1 = np.random.choice(np.arange(0,l1),(1,l1), replace=False)
            perm2 = np.random.choice(np.arange(0,l2),(1,l2), replace=False)
      
        sets=[]
        for i in range(0, k):
            sets.append([])
            
        i = 0
        while i<l1:
            for v in range(0,k):
                if i<l1:
                    sets[v].append(c1[0,perm1[0,i]])
                    i = i+1
        
        i = 0
        while i<l2:
            for v in range(0,k):
                if i<l2:
                    sets[v].append(c2[0,perm2[0,i]])
                    i = i+1

    return sets    