import numpy as np

def learn_error(y_learnt, y_test, learn_task):
    #LEARN_ERROR Compute the learning error.
    #   LRN_ERROR = LEARN_ERROR(Y_LEARNT, Y_TEST, LEARN_TASK) computes the 
    #   classification or regression error given two vectors 'Y_LEARNT' and 
    #   'T_TEST', which contain respectively the learnt and the the test 
    #   labels/values of the output set.
    #   The parameter 'LEARN_TASK' specify the kind of error:
    #       'regr' - regression error
    #       'class' - classification error
    #
    #   Example:
    #       y_learnt = Kernel * alpha;
    #       lrn_error = learn_error(y_learnt, y_test, 'class');
    #       lrn_error = learn_error(y_learnt, y_test, 'regr');
    #
    # See also LEARN
    y_learnt=np.reshape(y_learnt, (len(y_learnt),1))
    
    if learn_task=='Classification':
        lrn_error = (np.sum((np.multiply(y_learnt, y_test)) <= 0)) / float(len(y_test))
    elif learn_task=='Regression':
        lrn_error = ((np.linalg.norm((np.subtract(y_learnt, y_test)), ord=2))**2) / float(len(y_test))
    else:
        print 'Unknown learning task!'
    return lrn_error