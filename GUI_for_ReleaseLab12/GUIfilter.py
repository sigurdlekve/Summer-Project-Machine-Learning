'''
Created on 30. jun. 2016

@author: Sigurd Lekve
'''
import sys
sys.path.insert(0,"../ReleaseLab12_TRANSLATED")

# from KernelMatrix import KernelMatrix
# from MixGauss import MixGauss
# from regularizedKernLSTest import regularizedKernLSTest
# from regularizedKernLSTrain import regularizedKernLSTrain
# from separatingFKernRLS import separatingFKernRLS
# from SquareDist import SquareDist
# from flipLabels import flipLabels
# from two_moons import two_moons
# from holdoutCVKernRLS import holdoutCVKernRLS, calcErr

import numpy as np
from numpy import transpose as trsp
import matplotlib.pyplot as plt
from Tkinter import *

from MixGaussGUI import MixGaussGUI, MixGaussGUI_tr, MixGaussGUI_ts


root=Tk()

# Input Frame ****************************************************
input_frame=Frame(root, bg='yellow')
input_frame.pack(side=TOP, fill=X)

input_title=Label(input_frame, text='Input')
input_title.grid(row=0, column=0, sticky=W)

# Task Frame *****************************************************
task_frame=Frame(input_frame, bg='red')
task_frame.grid(row=1, column=0, sticky=W)

task_title=Label(task_frame, text='Task')
task_title.grid(row=0, column=0, sticky=W)

class_but_check=Checkbutton(task_frame, text='Classification')
class_but_check.grid(row=1, column=0, sticky=W)
reg_but_check=Checkbutton(task_frame, text='Regression', state=DISABLED)
reg_but_check.grid(row=1, column=1, sticky=E)

# Data Frame *****************************************************
data_frame=Frame(input_frame, bg='green')
data_frame.grid(row=2, column=0, sticky=W)

# ??? Existing dataset ???
data_title=Label(data_frame, text='Data type')
data_title.grid(row=0, column=0, sticky=W)

var_data=StringVar(data_frame)
var_data.set('Select data type')
data_type=OptionMenu(data_frame, var_data, 'Gaussian', 'Moons')
data_type.grid(row=1, column=1)

ntr_samples=Label(data_frame, text='# training samples')
ntr_samples.grid(row=2, column=0, sticky=E)
ntr_samples_entry=Entry(data_frame, bd=5, justify=CENTER)
ntr_samples_entry.grid(row=2, column=1)

nts_samples=Label(data_frame, text='# test samples')
nts_samples.grid(row=3, column=0, sticky=E)
nts_samples_entry=Entry(data_frame, bd=5, justify=CENTER)
nts_samples_entry.grid(row=3, column=1)

flip_ratio=Label(data_frame, text='Wrong label ratio')
flip_ratio.grid(row=4, column=0, sticky=E)
flip_ratio_entry=Entry(data_frame, bd=5, justify=CENTER)
flip_ratio_entry.grid(row=4, column=1)

def get_var_data():
    ntr=int(ntr_samples_entry.get())
    nts=int(nts_samples_entry.get())
    pflip=float(flip_ratio_entry.get())
    means=[[[0],[0]],[[1],[1]]]
    sigmas=[[0.5],[0.3]]
    return ntr, nts, pflip, means, sigmas
def MixGaussGUI():
    ntr, nts, pflip, means, sigmas=get_var_data()
    Xtr, Ytr = MixGaussGUI_tr(means, sigmas, ntr, pflip)
    Xts, Yts = MixGaussGUI_ts(means, sigmas, nts, pflip)
    plt.show()
    return Xtr, Ytr, Xts, Yts

load_data_but=Button(data_frame, text='Load data', command=MixGaussGUI)
load_data_but.grid(row=5, column=1, sticky=E)

#Filter Frame ****************************************************
filter_frame=Frame(root, bg='purple')
filter_frame.pack(side=TOP, fill=X)

filter_title=Label(filter_frame, text='Filter')
filter_title.grid(row=0, column=0, sticky=W)

var_filter=StringVar(filter_frame)
var_filter.set('Select filter')
filter_type=OptionMenu(filter_frame, var_filter, 'Reg. Least Squared')
filter_type.grid(row=1, column=0, sticky=W)

#Kernel Frame ****************************************************
kernel_frame=Frame(root, bg='blue')
kernel_frame.pack(side=TOP, fill=X)

kernel_title=Label(kernel_frame, text='Kernel')
kernel_title.grid(row=0, column=0, sticky=W)

var_kernel=StringVar(kernel_frame)
var_kernel.set('Select kernel')
kernel_type=OptionMenu(kernel_frame, var_kernel, 'Linear', 'Polynomial', 'Gaussian')
kernel_type.grid(row=1, column=0, sticky=W)

fixed_s_but_check=Checkbutton(kernel_frame, text='Use fixed s value')
fixed_s_but_check.grid(row=2, column=0, sticky=E)
fixed_s_entry=Entry(kernel_frame, bd=5, justify=CENTER)
fixed_s_entry.grid(row=2, column=1, sticky=E)

# ????? Choosing Sigma ??????
# ????? AutoSigma ??????

#Learning Frame *************************************************
learning_frame=Frame(root, bg='brown')
learning_frame.pack(side=TOP, fill=X)

learning_title=Label(learning_frame, text='Learning')
learning_title.grid(row=0, column=0, sticky=W)

# ????? Using KCV ??????

KCV_but_check=Checkbutton(learning_frame, text='Use KCV', state= DISABLED)
KCV_but_check.grid(row=1, column=0, sticky=W)

fixed_l_but_check=Checkbutton(learning_frame, text='Use fixed l value')
fixed_l_but_check.grid(row=2, column=0, sticky=E)
fixed_l_entry=Entry(learning_frame, bd=5, justify=CENTER)
fixed_l_entry.grid(row=2, column=1, sticky=E)

# Bottom Frame ***************************************************
bottom_frame=Frame(root, bg='pink')
bottom_frame.pack(side=BOTTOM, fill=X)

def get_var_l_s_kernel():
    KerPar=float(fixed_s_entry.get())
    l=float(fixed_l_entry.get()) 
    kernel=var_kernel.get()
    filter=var_kernel.get()
    return kernel, KerPar, l
def get_filter(X1, X2):
    if np.array_equal(filter, 'Reg. Least Squared'):
        filter=SquareDist(X1, X2)
    else:
        print 'Reg. Least Squared is only filter'
    return filter

def KernelMatrixGUI(X1, X2):
    kernel, KerPar, l = get_var_l_s_kernel()
    filter = get_filter(X1, X2)
    
    if np.size(kernel) == 0:
        kernel = 'linear'
    if np.array_equal(kernel, 'linear'):
        K = np.dot(X1,trsp(X2))
    elif np.array_equal(kernel, 'polynomial'):
        K = np.power((1 + np.dot(X1,trsp(X2))),param)
    elif np.array_equal(kernel, 'gaussian'):
        K = np.exp((float(-1)/(float(2*param**2)))*filter) 
    return K
  
def apply_classify():
    kernel, KerPar, l = get_l_s_kernel()
    c=regularizedKernLSTrain(Xtr, Ytr, kernel, KerPar, l)

run_but=Button(bottom_frame, text='Run')
run_but.pack(side=RIGHT)






root.mainloop()
