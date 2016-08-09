import numpy as np
from learn import learn
from learn_error import learn_error
from KernelMatrix import KernelMatrix, SquareDist

def create_classify_plot(alpha_best, Xtr, Ytr, Xts, Yts, kernel_type, s_value, task, step):
    
    min_ts0=min(Xts[:,0])
    min_tr0=min(Xtr[:,0])
    min_abs0=min([min_ts0, min_tr0])
      
    max_ts0=max(Xts[:,0])
    max_tr0=max(Xtr[:,0])
    max_abs0=max([max_ts0, max_tr0])
    
    min_ts1=min(Xts[:,1])
    min_tr1=min(Xtr[:,1])
    min_abs1=min([min_ts1, min_tr1])
      
    max_ts1=max(Xts[:,1])
    max_tr1=max(Xtr[:,1])
    max_abs1=max([max_ts1, max_tr1])
    
    ax=np.append(np.arange(min_abs0, max_abs0, step),max_abs0)
    az=np.append(np.arange(min_abs1, max_abs1, step),max_abs1)
    a, b = np.meshgrid(ax,az)
    na = np.reshape(a.T,(np.size(a),1))
    nb = np.reshape(b.T,(np.size(b),1))
    c = np.concatenate((na, nb), axis=1)
    
    K=KernelMatrix(c, Xtr, kernel_type, s_value)
    y_learnt=np.dot(K, alpha_best)
    z=np.array(np.reshape(y_learnt,(np.size(a,axis=1),np.size(a,axis=0))).T)
    z=np.array(z)
    return a, b, z