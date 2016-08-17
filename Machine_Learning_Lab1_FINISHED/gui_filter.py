'''
Created on 27. jul. 2016

@author: Sigurd Lekve
'''

import sys

import numpy as np
import math
import scipy
from numpy import transpose as trsp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Tkinter import *
import tkMessageBox
from ttk import *

from dataset_scripts import create_dataset

from dataset_scripts import gaussian
from dataset_scripts import linear_data
from dataset_scripts import moons
from dataset_scripts import sinusoidal
from dataset_scripts import spiral

from spectral_reg_toolbox import flipLabels
from spectral_reg_toolbox import EuclideanDistance
from spectral_reg_toolbox import autosigma
from spectral_reg_toolbox import holdoutCVKernRLS
from spectral_reg_toolbox import KernelMatrix
from spectral_reg_toolbox import regularizedKernLSTest
from spectral_reg_toolbox import regularizedKernLSTrain
from spectral_reg_toolbox import separatingFKernRLS
from spectral_reg_toolbox import learn
from spectral_reg_toolbox import learn_error
from spectral_reg_toolbox import splitting
from spectral_reg_toolbox import tsvd
from spectral_reg_toolbox import rls
from spectral_reg_toolbox import cutoff
from spectral_reg_toolbox import land
from spectral_reg_toolbox import nu
from spectral_reg_toolbox import kcv
from spectral_reg_toolbox import create_classify_plot



root=Tk()

#Setting corner icon and title for main window.
#root.iconbitmap('simula.ico')
root.wm_title('Machine Learning Lab')

#Selecting a global theme for all widgets.
s=Style()
s.theme_use('clam')

s.configure('New.TCheckbutton', font=('Verdana', 10))
s.configure('New.TLabel', font=('Verdana', 10))
s.configure('Title.TLabel', font=('Verdana', 13, 'bold'))
s.configure('New.TButton', font=('Verdana', 10))
#s.configure('New.TEntry', font=('Verdana', 16))
#s.configure('New.TOptionMenu', font=('Verdana', 10))



# Task Frame *****************************************************

#Creating task frame
task_frame=Frame(root)
#task_frame.grid(row=1, column=0, sticky=W)

task_title=Label(task_frame, text='Task', style='Title.TLabel')
task_title.grid(row=0, column=0, sticky=N+S+E+W)

def keep_task_check():
    check_v_class.set(1)

check_v_class=IntVar()
class_but_check=Checkbutton(task_frame, text='Classification', variable=check_v_class, command=keep_task_check, style='New.TCheckbutton')
check_v_class.set(1)
class_but_check.grid(row=1, column=0, sticky=N+S+E+W)

check_v_reg=IntVar()
reg_but_check=Checkbutton(task_frame, text='Regression', variable=check_v_reg, state=DISABLED, style='New.TCheckbutton')
check_v_reg.set(0)
reg_but_check.grid(row=1, column=1, sticky=N+S+E+W)




# Data Frame *****************************************************
data_frame=Frame(root)
#data_frame.grid(row=2, column=0, sticky=N+S+E+W)

# ??? Existing dataset ???
data_title=Label(data_frame, text='Data type', style='Title.TLabel')
data_title.grid(row=0, column=0, sticky=N+S+E+W)

var_data=StringVar(data_frame)
data_type_menu=OptionMenu(data_frame, var_data,'Gaussian', 'Gaussian', 'Moons', 'Spiral', 'Sinusoidal', 'Linear')
data_type_menu.grid(row=1, column=1)

ntr_samples=Label(data_frame, text='# training samples', style='New.TLabel')
ntr_samples.grid(row=2, column=0, sticky=N+S+E+W)
ntr_samples_entry=Entry(data_frame, justify=CENTER, font=('Verdana', 10))
ntr_samples_entry.insert(0,100)
ntr_samples_entry.grid(row=2, column=1)

nts_samples=Label(data_frame, text='# test samples', style='New.TLabel')
nts_samples.grid(row=3, column=0, sticky=N+S+E+W)
nts_samples_entry=Entry(data_frame, justify=CENTER, font=('Verdana', 10))
nts_samples_entry.insert(0, 5000)
nts_samples_entry.grid(row=3, column=1)

flip_ratio=Label(data_frame, text='Wrong label ratio', style='New.TLabel')
flip_ratio.grid(row=4, column=0, sticky=N+S+E+W)
flip_ratio_entry=Entry(data_frame, justify=CENTER, font=('Verdana', 10))
flip_ratio_entry.insert(0, 0.0)
flip_ratio_entry.grid(row=4, column=1)


def create_data():
    data_type=var_data.get()
    try:
        ntr=int(ntr_samples_entry.get())
        nts=int(nts_samples_entry.get())
        pflip=float(flip_ratio_entry.get())
        if pflip >= 1 or pflip<0 or ntr<0 or nts<0:
            check_data=tkMessageBox.showwarning('Tips and Tricks', '# of training and test samples must be a positive integer,\n and wrong label ratio must be less than one and greater\n or equal to zero.')
            return
    except ValueError:
        check_data=tkMessageBox.showwarning('Tips and Tricks', '# of training and test samples must be a positive integer,\n and wrong label ratio must be less than one and greater\n or equal to zero.')
        return
    Xtr, Ytr = create_dataset(ntr, data_type, 0.0)
    Xts, Yts = create_dataset(nts, data_type, 0.0)
    Ytr = flipLabels(Ytr, pflip)
    Yts = flipLabels(Yts, pflip)

    np.savez('loadeddata',Xtr=Xtr,Ytr=Ytr,Xts=Xts,Yts=Yts)
      
    subplot_ts.hold(False)
    subplot_tr.hold(False)
    subplot_ts.scatter(Xts[:,0],Xts[:,1],25,Yts,edgecolor='None')
    subplot_tr.scatter(Xtr[:,0],Xtr[:,1],25,Ytr,edgecolor='None')
    canvas=FigureCanvasTkAgg(main_plot, plot_frame)
    canvas.get_tk_widget().grid(row=1, column=0) 
      
    return Xtr, Ytr, Xts, Yts

load_data_but=Button(data_frame, text='Load data', command=create_data, style='New.TButton')
load_data_but.grid(row=5, column=1, sticky=N+S+E+W)






#Filter Frame ****************************************************
filter_frame=Frame(root)

filter_title=Label(filter_frame, text='Filter', style='Title.TLabel')
filter_title.grid(row=0, column=0, sticky=N+S+E+W)

def filter_choice(filter_type):
    if check_v_KCV.get()==1:
        filter_type=var_filter.get()
        if filter_type=='Landweber' or filter_type=='NU-method':
            tmin_entry.config(state='disabled')
            nt_values_entry.config(state='disabled')
            Space_type_menu.config(state='disabled')
        elif filter_type=='Reg. Least Squared' or filter_type=='Truncated SVD' or filter_type=='Spectral Cut-Off':
            tmin_entry.config(state='active')
            nt_values_entry.config(state='active')
            Space_type_menu.config(state='active')

var_filter=StringVar(filter_frame)
filter_type_menu=OptionMenu(filter_frame, var_filter, 'Reg. Least Squared', 'Reg. Least Squared', 'Truncated SVD', 'Spectral Cut-Off', 'Landweber', 'NU-method', command=filter_choice)
filter_type_menu.grid(row=1, column=0, sticky=N+S+E+W)

    
#Kernel Frame ****************************************************
kernel_frame=Frame(root)

kernel_title=Label(kernel_frame, text='Kernel', style='Title.TLabel')
kernel_title.grid(row=0, column=0, sticky=N+S+E+W)

def kernel_choice(kernel):
    kernel=var_kernel.get()
    if kernel=='Linear':
        check_v_fixed_s.set(0)
        check_v_auto.set(0)
        fixed_KerPar_entry.config(state='disabled')
        fixed_KerPar_but_check.config(state='disabled')
        autosigma_but_check.config(state='disabled')
    elif kernel=='Polynomial':
        check_v_fixed_s.set(1)
        check_v_auto.set(0)
        fixed_KerPar_entry.config(state='active')
        fixed_KerPar_but_check.config(state='active')
        autosigma_but_check.config(state='disabled')
    elif kernel=='Gaussian':
        check_v_fixed_s.set(1)
        check_v_auto.set(0)
        fixed_KerPar_entry.config(state='active')
        fixed_KerPar_but_check.config(state='active')
        autosigma_but_check.config(state='active')

def checking_auto():
    kernel=var_kernel.get()
    if check_v_auto.get()==1:
        check_v_fixed_s.set(0)
        fixed_KerPar_entry.config(state='disabled')
    elif check_v_auto.get()==0:
        check_v_auto.set(0)
        check_v_fixed_s.set(1)
        fixed_KerPar_entry.config(state='active')
        if kernel=='Polynomial':
            check_v_fixed_s.set(1)
    
def checking_fixed_KerPar():
    kernel=var_kernel.get()
    if kernel=='Polynomial':
        check_v_fixed_s.set(1)
    
    if check_v_fixed_s.get()==1:
        check_v_auto.set(0)
        fixed_KerPar_entry.config(state='active')
    elif check_v_fixed_s.get()==0:
        check_v_fixed_s.set(0)
        fixed_KerPar_entry.config(state='disabled')
                 

check_v_fixed_s=IntVar()
check_v_auto=IntVar()

var_kernel=StringVar(kernel_frame)
kernel_type_menu=OptionMenu(kernel_frame, var_kernel,'Polynomial', 'Linear', 'Polynomial', 'Gaussian', command=kernel_choice)
kernel_type_menu.grid(row=1, column=0, sticky=N+S+E+W)

fixed_KerPar_but_check=Checkbutton(kernel_frame, text='Use fixed s value', variable=check_v_fixed_s, command=checking_fixed_KerPar, style='New.TCheckbutton')
check_v_fixed_s.set(1)
fixed_KerPar_but_check.grid(row=2, column=0, sticky=N+S+E+W)
fixed_KerPar_entry=Entry(kernel_frame, justify=CENTER, font=('Verdana', 10))
fixed_KerPar_entry.insert(0,3)
fixed_KerPar_entry.grid(row=2, column=1, sticky=N+S+E+W)

autosigma_but_check=Checkbutton(kernel_frame, text='Autosigma', state=DISABLED, variable=check_v_auto, command=checking_auto, style='New.TCheckbutton')
check_v_auto.set(0)
autosigma_but_check.grid(row=3, column=0, sticky=N+S+E+W)

# ????? Choosing Sigma ??????
# ????? AutoSigma ??????





#Learning Frame *************************************************
learning_frame=Frame(root)

learning_title=Label(learning_frame, text='Learning', style='Title.TLabel')
learning_title.grid(row=0, column=0, sticky=N+S+E+W)

# ????? Using KCV ??????

def checking_KCV():
    if check_v_KCV.get()==1:
        check_v_fixed_t.set(0)
        fixed_t_entry.config(state='disabled')
        Split_type_menu.config(state='active')
        n_split_entry.config(state='active')
        tmin_entry.config(state='active')
        tmax_entry.config(state='active')
        nt_values_entry.config(state='active')
        Space_type_menu.config(state='active')
        filter_type=var_filter.get()
        filter_choice(filter_type)
    elif check_v_KCV.get()==0:
        check_v_fixed_t.set(1)
        Split_type_menu.config(state='disabled')
        n_split_entry.config(state='disabled')
        tmin_entry.config(state='disabled')
        tmax_entry.config(state='disabled')
        nt_values_entry.config(state='disabled')
        Space_type_menu.config(state='disabled')
        fixed_t_entry.config(state='active')
    
def checking_fixed_t():
    if check_v_fixed_t.get()==1:
        check_v_KCV.set(0)
        Split_type_menu.config(state='disabled')
        n_split_entry.config(state='disabled')
        tmin_entry.config(state='disabled')
        tmax_entry.config(state='disabled')
        nt_values_entry.config(state='disabled')
        Space_type_menu.config(state='disabled')
        fixed_t_entry.config(state='active')
    elif check_v_fixed_t.get()==0:
        check_v_KCV.set(1)
        fixed_t_entry.config(state='disabled')
        Split_type_menu.config(state='active')
        n_split_entry.config(state='active')
        tmin_entry.config(state='active')
        tmax_entry.config(state='active')
        nt_values_entry.config(state='active')
        Space_type_menu.config(state='active')
        filter_type=var_filter.get()
        filter_choice(filter_type)

check_v_KCV=IntVar()
check_v_fixed_t=IntVar()

fixed_t_but_check=Checkbutton(learning_frame, text='Use fixed t value', variable=check_v_fixed_t, command=checking_fixed_t, style='New.TCheckbutton')
check_v_fixed_t.set(1)
fixed_t_but_check.grid(row=8, column=0, sticky=N+S+E+W)

fixed_t_entry=Entry(learning_frame, justify=CENTER, font=('Verdana', 10))
fixed_t_entry.insert(0,0.01)
fixed_t_entry.grid(row=8, column=1, sticky=N+S+E+W)

KCV_but_check=Checkbutton(learning_frame, text='Use KCV', variable=check_v_KCV, command=checking_KCV, style='New.TCheckbutton')
check_v_KCV.set(0)
KCV_but_check.grid(row=1, column=0, sticky=N+S+E+W)

Split=Label(learning_frame, text='Split', style='New.TLabel')
Split.grid(row=2, column=0, sticky=N+S+E+W)
var_Split=StringVar(learning_frame)
Split_type_menu=OptionMenu(learning_frame, var_Split,'Sequential', 'Sequential', 'Random')
Split_type_menu.grid(row=2, column=1)

n_split=Label(learning_frame, text='# split', style='New.TLabel')
n_split.grid(row=3, column=0, sticky=N+S+E+W)
n_split_entry=Entry(learning_frame, justify=CENTER, font=('Verdana', 10))
n_split_entry.insert(0, 5)
n_split_entry.grid(row=3, column=1)

tmin_l=Label(learning_frame, text='t min', style='New.TLabel')
tmin_l.grid(row=4, column=0, sticky=N+S+E+W)
tmin_entry=Entry(learning_frame, justify=CENTER, font=('Verdana', 10))
tmin_entry.insert(0, 0.001)
tmin_entry.grid(row=4, column=1)

tmax_l=Label(learning_frame, text='t max', style='New.TLabel')
tmax_l.grid(row=5, column=0, sticky=N+S+E+W)
tmax_entry=Entry(learning_frame, justify=CENTER, font=('Verdana', 10))
tmax_entry.insert(0, 1)
tmax_entry.grid(row=5, column=1)

nt_values_l=Label(learning_frame, text='# of t values', style='New.TLabel')
nt_values_l.grid(row=6, column=0, sticky=N+S+E+W)
nt_values_entry=Entry(learning_frame, justify=CENTER, font=('Verdana', 10))
nt_values_entry.insert(0, 10)
nt_values_entry.grid(row=6, column=1)

Space=Label(learning_frame, text='Space', style='New.TLabel')
Space.grid(row=7, column=0, sticky=N+S+E+W)
var_Space=StringVar(learning_frame)
Space_type_menu=OptionMenu(learning_frame, var_Space,'Linear space', 'Linear space', 'Log space')
Space_type_menu.grid(row=7, column=1)

Split_type_menu.config(state='disabled')
n_split_entry.config(state='disabled')
tmin_entry.config(state='disabled')
tmax_entry.config(state='disabled')
nt_values_entry.config(state='disabled')
Space_type_menu.config(state='disabled')






#Results Frame ******************************************************
results_frame=Frame(root)

results_title=Label(results_frame, text='Results', style='Title.TLabel')
results_title.grid(row=0, column=0, sticky=N+S+E+W)

tr_error=' - - - '
ts_error=' - - - '
select_t=' - - - '
select_sigma=' - - - '
eplot_xlabel_t=' '

tr_error_title=Label(results_frame, text='Training error', style='New.TLabel')
tr_error_title.grid(row=1, column=0, sticky=N+S+E+W)
tr_error_v=Label(results_frame, text=tr_error, style='New.TLabel')
tr_error_v.grid(row=1, column=2, sticky=N+S+E+W)

ts_error_title=Label(results_frame, text='Test error', style='New.TLabel')
ts_error_title.grid(row=2, column=0, sticky=N+S+E+W)
ts_error_v=Label(results_frame, text=ts_error, style='New.TLabel')
ts_error_v.grid(row=2, column=2, sticky=N+S+E+W)

select_t_title=Label(results_frame, text='Selected t', style='New.TLabel')
select_t_title.grid(row=3, column=0, sticky=N+S+E+W)
select_t_v=Label(results_frame, text=select_t, style='New.TLabel')
select_t_v.grid(row=3, column=2, sticky=N+S+E+W)

select_sigma_title=Label(results_frame, text='Selected sigma', style='New.TLabel')
select_sigma_title.grid(row=4, column=0, sticky=N+S+E+W)
select_sigma_v=Label(results_frame, text=select_sigma, style='New.TLabel')
select_sigma_v.grid(row=4, column=2, sticky=N+S+E+W)

def change_results(tr_error, ts_error, select_t, select_sigma, eplot_xlabel_t):
    tr_error=str(tr_error)
    ts_error=str(ts_error)
    select_t=str(select_t)
    select_sigma=str(select_sigma)
    tr_error_v.config(text=tr_error)
    ts_error_v.config(text=ts_error)
    select_t_v.config(text=select_t)
    select_sigma_v.config(text=select_sigma)
    
    eplot_xlabel_t=str(eplot_xlabel_t)
    eplot_xlabel.config(text=eplot_xlabel_t)
    
    return





#Eplot_frame Frame **************************************************
eplot_frame=Frame(root)

eplot_xlabel=Label(eplot_frame, text=eplot_xlabel_t, style='New.TLabel')
eplot_xlabel.grid(row=4, column=0, sticky=N+S+E+W)

error_plot=Figure(figsize=(6,4), facecolor='lightgrey')

subplot_error_t=error_plot.add_subplot(1,1,1)
subplot_error_t.set_title('KCV - Error plot')
subplot_error_t.set_ylabel('Error')
e_empty1=0
e_empty2=0
subplot_error_t.plot(e_empty1, e_empty2)

e_canvas=FigureCanvasTkAgg(error_plot, eplot_frame)          
e_canvas.get_tk_widget().grid(columnspan=2, row=0, column=0)

check_v_logx=IntVar()

ax_logx_check=Checkbutton(eplot_frame, text='Logaritmic x-axis', variable=check_v_logx, style='New.TCheckbutton')
check_v_logx.set(0)
ax_logx_check.grid(row=3, column=0, sticky=N+S+E+W)

#Plot Frame *****************************************************
plot_frame=Frame(root)

plot_title=Label(plot_frame, text='Plot', style='Title.TLabel')
plot_title.grid(row=0, column=0, sticky=N+S+E+W)

main_plot=Figure(figsize=(14,5), facecolor='lightgrey')

subplot_ts=main_plot.add_subplot(1,2,1)
subplot_ts.set_title('Test')
subplot_tr=main_plot.add_subplot(1,2,2)
subplot_tr.set_title('Training')
empty1=0
empty2=0
subplot_ts.plot(empty1, empty2)
subplot_tr.plot(empty1, empty2)

subplot_classify_ts=main_plot.add_subplot(1,2,1)
subplot_classify_tr=main_plot.add_subplot(1,2,2)
empty1c=0
empty2c=0
subplot_classify_ts.plot(empty1c, empty2c)
subplot_classify_tr.plot(empty1c, empty2c)


canvas=FigureCanvasTkAgg(main_plot, plot_frame)          
canvas.get_tk_widget().grid(row=1, column=0)    




# Bottom Frame ***************************************************
bottom_frame=Frame(root)

def apply_classify():
    
    subplot_ts.hold(False)
    subplot_tr.hold(False)
    subplot_classify_ts.hold(False)
    subplot_classify_tr.hold(False)    
    
    try:
        datafile = np.load('loadeddata.npz')
    except IOError:
        tkMessageBox.showwarning('Tips and Tricks', 'You have to load data before running.')
        return
    
    Xtr=datafile['Xtr']
    Ytr=datafile['Ytr']
    Xts=datafile['Xts']
    Yts=datafile['Yts']

    #Getting parameter from GUI:
        
    #Getting kernel type.    
    kernel_type=var_kernel.get()
    
    #Getting sigma value (fixed or auto).
    if check_v_fixed_s.get()==1:
        s_value=float(fixed_KerPar_entry.get())
    elif check_v_auto.get()==1:
        s_value=float(autosigma(Xtr, 5))
    #Checking sigma input for kernel:
    if kernel_type=='Polynomial':
        try:
            s_value=int(s_value)
            if s_value<=0:
               check_s_pol=tkMessageBox.showwarning('Tips and Tricks', 'The degree of the polynomial kernel has to be an integer greater than zero.')
               return
        except ValueError:
            check_s_pol=tkMessageBox.showwarning('Tips and Tricks', 'The degree of the polynomial kernel has to be an integer greater than zero.')
            return
    elif kernel_type=='Gaussian':
        if s_value<=0:
            check_s_gauss=tkMessageBox.showwarning('Tips and Tricks', 'The kernel parameter for the gaussian kernel has to be greater than zero.')
            return
    elif kernel_type=='Linear':
        s_value=[] #Setting sigma value to empty for linear kernel.
         
    
    #Getting task (Classification or Regression).
    if check_v_class.get()==1:
        task='Classification'
    elif check_v_class.get()==1:
        task='Regression'    

    #Running with fixed regularization parameter (t).    
    if check_v_fixed_t.get()==1:
        #Getting parameter from GUI:
        
        #Getting filter type.
        filter_type=var_filter.get()
        
        #Getting fixed regularization prameter (t).
        trange=[float(fixed_t_entry.get())]
        
        #Checking t value for filters:
        if filter_type=='Landweber':
            trange=[fixed_t_entry.get()]
            try:
                trange=[int(trange[0])]
                if trange[0]<=1:
                    check_tval_land=tkMessageBox.showwarning('Tips and Tricks', 't max or fixed t must be a positive integer greater than one\n when using the filter Landweber.')
                    return
            except ValueError:
                check_tval_land=tkMessageBox.showwarning('Tips and Tricks', 't max or fixed t must be a positive integer greater than one\n when using the filter Landweber.')
                return
        elif filter_type=='NU-method':
            trange=[fixed_t_entry.get()]
            try:
                trange=[int(trange[0])]
                if trange[0]<=2:
                    check_tval_nu=tkMessageBox.showwarning('Tips and Tricks', 't max or fixed t must be a positive integer greater than two\n when using the filter NU-method.')
                    return
            except ValueError:
                check_tval_nu=tkMessageBox.showwarning('Tips and Tricks', 't max or fixed t must be a positive integer greater than two\n when using the filter NU-method.')
                return
        elif filter_type=='Reg. Least Squared' or filter_type=='Truncated SVD' or filter_type=='Spectral Cut-Off':
            trange=[fixed_t_entry.get()]
            try:
                trange=[float(trange[0])]
                if trange[0]<=0:
                    check_tval_other=tkMessageBox.showwarning('Tips and Tricks', 'Fixed t must be a positive number greater than zero.')
                    return
            except ValueError:
                check_tval_other=tkMessageBox.showwarning('Tips and Tricks', 'Fixed t must be a positive number greater than zero.')
                return
        
        #Plotting the classifier with the given parameters.
        alpha, err = learn(kernel_type, s_value, filter_type, trange, Xtr, Ytr, task)
        
        if filter_type=='Landweber' or filter_type=='NU-method':
            min_err=min(err[0])
            index=np.argmin(err[0])
        elif filter_type=='Reg. Least Squared' or filter_type=='Truncated SVD' or filter_type=='Spectral Cut-Off':
            min_err = min(err)
            index = np.argmin(err)
        
        #Get best coefficients
        alpha_best = alpha[:, index]
        
        #Calculating error on test set
        K=KernelMatrix(Xts, Xtr, kernel_type, s_value)
        y_learnt = np.dot(K, alpha[:,index])
        lrn_error_ts = learn_error(y_learnt, Yts, task)
        
        step=0.1
        a, b, z = create_classify_plot(alpha_best, Xtr, Ytr, Xts, Yts, kernel_type, s_value, step)
        
        subplot_classify_ts.contourf(a, b, z, 1, colors=('w', 'g'), alpha=.3, antialiased=True)
        subplot_classify_tr.contourf(a, b, z, 1, colors=('w', 'g'), alpha=.3, antialiased=True)
        canvas=FigureCanvasTkAgg(main_plot, plot_frame)
        canvas.get_tk_widget().grid(row=1, column=0)
        subplot_ts.hold(True)
        subplot_tr.hold(True)
        subplot_ts.scatter(Xts[:,0],Xts[:,1],25,Yts,edgecolor='None')
        subplot_tr.scatter(Xtr[:,0],Xtr[:,1],25,Ytr,edgecolor='None')
        canvas=FigureCanvasTkAgg(main_plot, plot_frame)
        canvas.get_tk_widget().grid(row=1, column=0)
        
        
        select_t=float(trange[0])
        if kernel_type=='Linear':
            select_sigma=' - - - '
        else:
            select_sigma=float(s_value) 
        ts_error=float(lrn_error_ts)
        tr_error=float(min_err)
        eplot_xlabel_t=' '
        
        change_results(tr_error, ts_error, select_t, select_sigma, eplot_xlabel_t)
        
    elif check_v_KCV.get()==1:

        subplot_error_t.hold(False)
        
        #Getting parameters from GUI:
        
        #Getting parameters for the range of regularization parameters used in the KCV-method.    
        
        #Checking tmin, tmax, nt_values:
        filter_type=var_filter.get()
        
        if filter_type=='Reg. Least Squared' or filter_type=='Truncated SVD' or filter_type=='Spectral Cut-Off':
            try:
                tmin=float(tmin_entry.get())
                tmax=float(tmax_entry.get())
                nt_values=int(nt_values_entry.get())
                if tmin>=tmax or tmin==0:
                    check_minmax=tkMessageBox.showwarning('Tips and Tricks', 'tmin has to be less than tmax, and cant be zero.')
                    return
                elif tmin<=0 or tmax<=0 or nt_values<=0:
                    check_minmax=tkMessageBox.showwarning('Tips and Tricks', 'tmin and tmax have to be positive numbers,\n and number of t values has to be an positive integer.')
                    return
            except ValueError:
                check_trange_input=tkMessageBox.showwarning('Tips and Tricks', 'tmin and tmax have to be positive numbers,\n and number of t values has to be an positive integer.')
                return
        elif filter_type=='Landweber':
            trange=tmax_entry.get()
            try:
                trange=[int(trange)]
                if trange[0]<=1:
                    check_tval_land=tkMessageBox.showwarning('Tips and Tricks', 't max or fixed t must be a positive integer greater than one\n when using the filter Landweber.')
                    return
            except ValueError:
                check_tval_land=tkMessageBox.showwarning('Tips and Tricks', 't max or fixed t must be a positive integer greater than one\n when using the filter Landweber.')
                return
        elif filter_type=='NU-method':
            trange=tmax_entry.get()
            try:
                trange=[int(trange)]
                if trange[0]<=2:
                    check_tval_nu=tkMessageBox.showwarning('Tips and Tricks', 't max or fixed t must be a positive integer greater than two\n when using the filter NU-method.')
                    return
            except ValueError:
                check_tval_nu=tkMessageBox.showwarning('Tips and Tricks', 't max or fixed t must be a positive integer greater than two\n when using the filter NU-method.')
                return
        
        #Getting the users choice of either linear space or logaritmic space for the regularization parameters used in the KCV-method.
        space_type=var_Space.get()
        if filter_type=='Reg. Least Squared' or filter_type=='Truncated SVD' or filter_type=='Spectral Cut-Off':
            if space_type=='Linear space':
                trange=np.linspace(tmin, tmax, nt_values)
            elif space_type=='Log space':
                trange=np.logspace(np.log(tmin), np.log(tmax), nt_values)
            
        #Getting the type of split and the number of splits to be used on the data set in the KCV-method (Sequential or Random).    
        split_type=var_Split.get()
        try:
            k=float(int(n_split_entry.get()))
            if k<=0:
                check_k=tkMessageBox.showwarning('Tips and Tricks', 'Number of splits has to be a positive integer.')
                return
        except ValueError:
            check_k=tkMessageBox.showwarning('Tips and Tricks', 'Number of splits has to be a positive integer.')
            return
        
        #Using KCV-method for choosing regularization parameter
        t_kcv_idx, avg_err_kcv=kcv(kernel_type, s_value, filter_type, trange, Xtr, Ytr, k, task, split_type)
        avg_err_kcv=np.ravel(avg_err_kcv)
        
        if filter_type=='Landweber' or filter_type=='NU-method':
            tval=[t_kcv_idx+1]
#             if filter_type == 'NU-method':
#                 while tval[0]<3:
#                     tval[0]=tval[0]+1
        elif filter_type=='Reg. Least Squared' or filter_type=='Truncated SVD' or filter_type=='Spectral Cut-Off':
            tval=[trange[t_kcv_idx]]
        
        alpha, err = learn(kernel_type, s_value, filter_type, tval, Xtr, Ytr, task)
        
        if filter_type=='Landweber' or filter_type=='NU-method':
            min_err=min(err[0])
            index=np.argmin(err[0])
        elif filter_type=='Reg. Least Squared' or filter_type=='Truncated SVD' or filter_type=='Spectral Cut-Off':
            min_err = min(err)
            index = np.argmin(err)
        
        #Get best coefficients
        alpha_best = alpha[:, index]
        
        #Calculating error on test set
        K=KernelMatrix(Xts, Xtr, kernel_type, s_value)
        y_learnt = np.dot(K, alpha[:,index])
        lrn_error_ts = learn_error(y_learnt, Yts, task)
        
        #Plotting the classifier
        step=0.1
        a, b, z = create_classify_plot(alpha_best, Xtr, Ytr, Xts, Yts, kernel_type, s_value, step)
        
        subplot_classify_ts.contourf(a, b, z, 1, colors=('w', 'g'), alpha=.3, antialiased=True)
        subplot_classify_tr.contourf(a, b, z, 1, colors=('w', 'g'), alpha=.3, antialiased=True)
        canvas=FigureCanvasTkAgg(main_plot, plot_frame)
        canvas.get_tk_widget().grid(row=1, column=0)
        subplot_ts.hold(True)
        subplot_tr.hold(True)
        subplot_ts.scatter(Xts[:,0],Xts[:,1],25,Yts,edgecolor='None')
        subplot_tr.scatter(Xtr[:,0],Xtr[:,1],25,Ytr,edgecolor='None')
        canvas=FigureCanvasTkAgg(main_plot, plot_frame)
        canvas.get_tk_widget().grid(row=1, column=0)
        
        if filter_type=='Reg. Least Squared' or filter_type=='Truncated SVD' or filter_type=='Spectral Cut-Off':
            subplot_error_t.plot(trange, avg_err_kcv, 'b--')
            eplot_xlabel_t='reg.par range: %.4f - %.4f' %(min(trange), max(trange))
            subplot_error_t.hold(True)
            subplot_error_t.grid(True)
            subplot_error_t.plot(trange[t_kcv_idx], avg_err_kcv[t_kcv_idx], 'ro', linewidth=10.0, label='Test error')
            subplot_error_t.hold(True)
            subplot_error_t.set_title('KCV - Error plot')
            e_canvas=FigureCanvasTkAgg(error_plot, eplot_frame)  
            e_canvas.get_tk_widget().grid(columnspan=2, row=0, column=0)
        elif filter_type=='Landweber' or filter_type=='NU-method':
            plot_trange= np.arange(1, len(avg_err_kcv)+1)
            subplot_error_t.plot(plot_trange, avg_err_kcv, 'b--')
            eplot_xlabel_t='t range: 0 - %.4f' %float(trange[0])
            subplot_error_t.hold(True)
            subplot_error_t.grid(True)
            subplot_error_t.plot(tval[0], avg_err_kcv[t_kcv_idx], 'ro', linewidth=10.0, label='Test error')
            subplot_error_t.hold(True)
            subplot_error_t.set_title('KCV - Error plot')
            e_canvas=FigureCanvasTkAgg(error_plot, eplot_frame)          
            e_canvas.get_tk_widget().grid(columnspan=2, row=0, column=0)
            
        select_t=float(tval[0])
        if kernel_type=='Linear':
            select_sigma=' - - - '
        else:
            select_sigma=float(s_value)
        ts_error=float(lrn_error_ts)
        tr_error=float(min_err)
        
        if check_v_logx.get()==1:
            subplot_error_t.set_xscale('log')    
        elif check_v_logx.get()==0:
            subplot_error_t.set_xscale('linear')
        
        change_results(tr_error, ts_error, select_t, select_sigma, eplot_xlabel_t)
    
    
    return

running=Toplevel(root)
run_label=Label(running, text='Computing...', font=('Verdana', 14))
run_label.pack(fill=BOTH, expand=True, padx=0, pady=0, ipadx=10, ipady=10)
running.state('withdrawn')


def button_push():
    running.state('normal')
    running.lift()
    running.update()
    apply_classify()
    running.withdraw()
     
    return 

run_but=Button(eplot_frame, text='Run', command=button_push, style='New.TButton')
run_but.grid(row=5, column=0, sticky=N+S+W+E)

# Placement ****************************************************
for root_cols in range(3):
  Grid.columnconfigure(root, root_cols, weight=1)
for root_rows in range(6):
  Grid.rowconfigure(root, root_rows, weight=1)


task_frame.grid(row=0, column=0, padx=3, pady=3, sticky=N+S+E+W)
data_frame.grid(row=1, column=0,padx=3, pady=3, sticky=N+S+E+W)
filter_frame.grid(row=2, column=0,padx=3, pady=3, sticky=N+S+E+W)
kernel_frame.grid(row=3, column=0,padx=3, pady=3, sticky=N+S+E+W)
learning_frame.grid(rowspan=2, row=0, column=1,padx=3, pady=3, sticky=N+S+E+W)
results_frame.grid(rowspan=2, row=2, column=1, padx=3, pady=3, sticky=N+S+E+W)
eplot_frame.grid(rowspan=4, row=0, column=2, padx=3, pady=3, sticky=N+S+E+W)
plot_frame.grid(row=5, columnspan=3,padx=3, pady=3, sticky=N+S+E+W)
bottom_frame.grid(row=4, columnspan=3,padx=3, pady=3, sticky=N+S+E+W)

#Task Frame
for task_cols in range(10):
  Grid.columnconfigure(task_frame, task_cols, weight=1)
for task_rows in range(10):
  Grid.rowconfigure(task_frame, task_rows, weight=1)
  
#Data Frame
for data_cols in range(10):
  Grid.columnconfigure(data_frame, data_cols, weight=1)
for data_rows in range(10):
  Grid.rowconfigure(data_frame, data_rows, weight=1)
  
#Filter Frame
for filter_cols in range(10):
  Grid.columnconfigure(filter_frame, filter_cols, weight=1)
for filter_rows in range(10):
  Grid.rowconfigure(filter_frame, filter_rows, weight=1)
  
#Kernel Frame
for kernel_cols in range(10):
  Grid.columnconfigure(kernel_frame, kernel_cols, weight=1)
for kernel_rows in range(10):
  Grid.rowconfigure(kernel_frame, kernel_rows, weight=1)
  
#Learning Frame
for learning_cols in range(10):
  Grid.columnconfigure(learning_frame, learning_cols, weight=1)
for learning_rows in range(10):
  Grid.rowconfigure(learning_frame, learning_rows, weight=1)
  
#Results Frame
for results_cols in range(10):
  Grid.columnconfigure(results_frame, results_cols, weight=1)
for results_rows in range(10):
  Grid.rowconfigure(results_frame, results_rows, weight=1)
  
#Eplot Frame
for eplot_cols in range(10):
  Grid.columnconfigure(eplot_frame, eplot_cols, weight=1)
for eplot_rows in range(10):
  Grid.rowconfigure(eplot_frame, eplot_rows, weight=1)
  
#Plot Frame
for plot_cols in range(1):
  Grid.columnconfigure(plot_frame, plot_cols, weight=1)
for plot_rows in range(2):
  Grid.rowconfigure(plot_frame, plot_rows, weight=1)
  
#Bottom Frame
for bottom_cols in range(10):
  Grid.columnconfigure(bottom_frame, bottom_cols, weight=1)
for bottom_rows in range(10):
  Grid.rowconfigure(bottom_frame, bottom_rows, weight=1)
  
# *****************************************************************
import os
import platform

def closing_action():
    if platform.system() == 'Windows':
        os.system('DEL loadeddata.npz')
    else:
        os.system('rm loadeddata.npz')
        
    root.destroy()  

root.protocol('WM_DELETE_WINDOW', closing_action)


root.mainloop()
