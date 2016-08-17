import numpy as np

def save_challenge_2(YOURNAME, T, KERNEL, KERNEL_PARAMETER, errval, FILT):
    #save_challenge_2(YOURNAME, T, KERNEL, KERNEL_PARAMETER, errval)
    #where YOURNAME is a string of the form 'name_surname'
    #T is the regularization pamareter you chose
    #KERNEL is a string and can be 'Linear', 'Polynomial', 'Gaussian'
    #KERNEL_PARAMETER is the parameter of the kernel
    #errval the mean average error aon the training set
    
    model=[]
    model.append([YOURNAME])
    model.append(T)
    model.append([KERNEL])
    model.append([KERNEL_PARAMETER])
    model.append(errval)
    model.append([FILT])
    np.savez(YOURNAME, model=model)
    
    for i in range(0,3):
        print ('%s you learned with success!\n\nYour learned model uses %s filter, %s kernel, kernel parameter %f and regularization'+\
                ' parameter %f. \n\nIts average kcv classification error is %.4f \n\n'+\
                'AND REMEMBER: if you like it submit your solution!!') % (YOURNAME, FILT, KERNEL, KERNEL_PARAMETER,T[i],errval[i])
    return