import numpy as np
from KernelMatrix import KernelMatrix, SquareDist

def patt_rec(knl, kpar, alpha, x_train, x_test, y_test, learn_task):
    #PATT_REC Calculates a test error given a training and a test dataset and an estimator
    #   [Y_LRNT, TEST_ERR] = PATT_REC(KNL, KPAR, ALPHA, X_TRAIN, X_TEST, Y_TEST)
    #   calculates the output vector Y_LRNT and the test error (regression or
    #   classification) TEST_ERR given
    #
    #      - kernel type specified by 'knl':
    #        'Linear'   - linear kernel, 'kpar' is not considered
    #        'Polynomial'   - polinomial kernel, where 'kpar' is the polinomial degree
    #        'Gaussian' - gaussian kernel, where 'kpar' is the gaussian sigma
    #
    #      - kernel parameter 'kpar':
    #        'deg'  - for polynomial kernel 'pol'
    #        'sigma' - for gaussian kernel 'gauss'
    #
    #      - an estimator 'alpha'
    #        training set 'x_train'
    #        test set 'x_test'
    #        known output labels/test data 'y_test'
    #
    #      - a learn_task 'learn_task'
    #        'Classification' - for classification
    #        'Regression' - for regression
    #    
    #   Example:
    #    y_lrnt, test_err = patt_rec('Gaussian', .4, alpha,x, x_test, y_test)
    #
    # See also LEARN, KERNEL, LEARN_ERROR
    
    K_test = KernelMatrix(x_test, x_train, knl, kpar) # Compute test kernel
    y_lrnt = np.dot(K_test, alpha)  # Computre predicted output vector
    
    test_err = learn_error(y_lrnt, y_test, learn_task) # Evaluate error
    return y_lrnt, test_err
