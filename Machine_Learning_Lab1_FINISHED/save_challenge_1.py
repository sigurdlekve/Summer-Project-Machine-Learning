import numpy as np

def save_challenge_1(YOURNAME, T, KERNEL, KERNEL_PARAMETER, errval):
    #save_challenge_1(YOURNAME, T, KERNEL, KERNEL_PARAMETER, errval)
    #where YOURNAME is a string of the form 'name_surname'
    #T is the regularization pamareter you chose
    #KERNEL is a string and can be 'Linear', 'Polynomial', 'Gaussian'
    #KERNEL_PARAMETER is the parameter of the kernel
    #errval the mean average error aon the training set
    
    model=[]
    model.append(YOURNAME)
    model.append(T)
    model.append(KERNEL)
    model.append(KERNEL_PARAMETER)
    model.append(errval)
    np.savez(YOURNAME, model=model)
    
    print ('%s you learned with success!\n\nYour learned model using %s kernel, kernel parameter %f and regularization'+\
           ' parameter %f. \n\nIts average kcv classification error is %.4f \n\n'+\
           'AND REMEMBER: if you like it submit your solution!!') % (YOURNAME, KERNEL, KERNEL_PARAMETER,T,errval)
    return