'''
Created on 27. jul. 2016

@author: Sigurd Lekve
'''

import sys

import numpy as np
import math
from numpy import transpose as trsp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Tkinter import *
import tkMessageBox
from ttk import *

#from dataGeneration import create_data
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



root=Tk()

root.iconbitmap('simula.ico')
root.wm_title('Machine Learning Lab')

s=Style()
s.theme_use('clam')

# Input Frame ****************************************************
#input_frame=Frame(root)

#input_title=Label(input_frame, text='Input', font=('Verdana', 12, 'bold'))
#input_title.grid(row=0, column=0, sticky=W)


# Task Frame *****************************************************
task_frame=Frame(root)
#task_frame.grid(row=1, column=0, sticky=W)

task_title=Label(task_frame, text='Task', font=('Verdana', 12, 'bold'))
task_title.grid(row=0, column=0, sticky=N+S+E+W)

def keep_task_check():
    check_v_class.set(1)

check_v_class=IntVar()
class_but_check=Checkbutton(task_frame, text='Classification', variable=check_v_class, command=keep_task_check)
check_v_class.set(1)
class_but_check.grid(row=1, column=0, sticky=N+S+E+W)

check_v_reg=IntVar()
reg_but_check=Checkbutton(task_frame, text='Regression', variable=check_v_reg, state=DISABLED)
check_v_reg.set(0)
reg_but_check.grid(row=1, column=1, sticky=N+S+E+W)




# Data Frame *****************************************************
data_frame=Frame(root)
#data_frame.grid(row=2, column=0, sticky=N+S+E+W)

# ??? Existing dataset ???
data_title=Label(data_frame, text='Data type', font=('Verdana', 12, 'bold'))
data_title.grid(row=0, column=0, sticky=N+S+E+W)

var_data=StringVar(data_frame)
data_type_menu=OptionMenu(data_frame, var_data,'Select data type', 'Gaussian', 'Moons', 'Spiral', 'Sinusoidal', 'Linear')
data_type_menu.grid(row=1, column=1)

ntr_samples=Label(data_frame, text='# training samples')
ntr_samples.grid(row=2, column=0, sticky=N+S+E+W)
ntr_samples_entry=Entry(data_frame, justify=CENTER)
ntr_samples_entry.insert(0,100)
ntr_samples_entry.grid(row=2, column=1)

nts_samples=Label(data_frame, text='# test samples')
nts_samples.grid(row=3, column=0, sticky=N+S+E+W)
nts_samples_entry=Entry(data_frame, justify=CENTER)
nts_samples_entry.insert(0, 5000)
nts_samples_entry.grid(row=3, column=1)

flip_ratio=Label(data_frame, text='Wrong label ratio')
flip_ratio.grid(row=4, column=0, sticky=N+S+E+W)
flip_ratio_entry=Entry(data_frame, justify=CENTER)
flip_ratio_entry.insert(0, 0.0)
flip_ratio_entry.grid(row=4, column=1)

def create_data():
    data_type=var_data.get()
    ntr=int(ntr_samples_entry.get())
    ntr1=int(math.ceil(ntr/2.0))
    ntr2=int(math.floor(ntr/2.0))
    nts=int(nts_samples_entry.get())
    nts1=int(math.ceil(float(nts/2)))
    nts2=int(math.floor(float(nts/2)))
    pflip=float(flip_ratio_entry.get())
  
    if data_type=='Gaussian':
        gaussian_input_tr=[ntr1, ntr2]
        gaussian_input_ts=[nts1, nts2]
        Xtr, Ytr = gaussian(gaussian_input_tr)
        Xts, Yts = gaussian(gaussian_input_ts)
        Ytr=flipLabels(Ytr, pflip)
        Yts=flipLabels(Yts, pflip)
    elif data_type=='Moons':
        moons_input_tr=[ntr1, ntr2]
        moons_input_ts=[nts1, nts2]
        Xtr, Ytr = moons(moons_input_tr)
        Xts, Yts = moons(moons_input_ts)
        Ytr=flipLabels(Ytr, pflip)
        Yts=flipLabels(Yts, pflip)
    elif data_type=='Spiral':
        spiral_input_tr=[ntr1, ntr2]
        spiral_input_ts=[nts1, nts2]
        Xtr, Ytr = spiral(spiral_input_tr)
        Xts, Yts = spiral(spiral_input_ts)
        Ytr=flipLabels(Ytr, pflip)
        Yts=flipLabels(Yts, pflip)
    elif data_type=='Sinusoidal':
        sinusoidal_input_tr=[ntr1, ntr2]
        sinusoidal_input_ts=[nts1, nts2]
        Xtr, Ytr = sinusoidal(sinusoidal_input_tr)
        Xts, Yts = sinusoidal(sinusoidal_input_ts)
        Ytr=flipLabels(Ytr, pflip)
        Yts=flipLabels(Yts, pflip)
    elif data_type=='Linear':
        linear_data_input_tr=[ntr1, ntr2]
        linear_data_input_ts=[nts1, nts2]
        Xtr, Ytr = linear_data(linear_data_input_tr)
        Xts, Yts = linear_data(linear_data_input_ts)
        Ytr=flipLabels(Ytr, pflip)
        Yts=flipLabels(Yts, pflip)
    np.savez('loadeddata',Xtr=Xtr,Ytr=Ytr,Xts=Xts,Yts=Yts)
      
    subplot_ts.hold(False)
    subplot_tr.hold(False)
    subplot_ts.scatter(Xts[:,0],Xts[:,1],25,Yts,edgecolor='None')
    subplot_tr.scatter(Xtr[:,0],Xtr[:,1],25,Ytr,edgecolor='None')
    canvas=FigureCanvasTkAgg(main_plot, plot_frame)
    canvas.get_tk_widget().grid(row=1, column=0) 
      
    return Xtr, Ytr, Xts, Yts

load_data_but=Button(data_frame, text='Load data', command=create_data)
load_data_but.grid(row=5, column=1, sticky=N+S+E+W)






#Filter Frame ****************************************************
filter_frame=Frame(root)

filter_title=Label(filter_frame, text='Filter', font=('Verdana', 12, 'bold'))
filter_title.grid(row=0, column=0, sticky=N+S+E+W)

def filter_choice(filter_type):
    if check_v_KCV.get()==1:
        filter_type=var_filter.get()
        if filter_type=='Landweber' or filter_type=='NU-method':
            tmin_l.config(state='disabled')
            tmin_entry.config(state='disabled')
            nt_values_l.config(state='disabled')
            nt_values_entry.config(state='disabled')
            Space_type_menu.config(state='disabled')
        elif filter_type=='Reg. Least Squared' or filter_type=='Truncated SVD' or filter_type=='Spectral Cut-Off':
            tmin_l.config(state='active')
            tmin_entry.config(state='active')
            nt_values_l.config(state='active')
            nt_values_entry.config(state='active')
            Space_type_menu.config(state='active')

var_filter=StringVar(filter_frame)
filter_type_menu=OptionMenu(filter_frame, var_filter, 'Select filter', 'Reg. Least Squared', 'Truncated SVD', 'Spectral Cut-Off', 'Landweber', 'NU-method', command=filter_choice)
filter_type_menu.grid(row=1, column=0, sticky=N+S+E+W)

    
#Kernel Frame ****************************************************
kernel_frame=Frame(root)

kernel_title=Label(kernel_frame, text='Kernel', font=('Verdana', 12, 'bold'))
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
kernel_type_menu=OptionMenu(kernel_frame, var_kernel,'Select kernel', 'Linear', 'Polynomial', 'Gaussian', command=kernel_choice)
kernel_type_menu.grid(row=1, column=0, sticky=N+S+E+W)

fixed_KerPar_but_check=Checkbutton(kernel_frame, text='Use fixed s value', variable=check_v_fixed_s, command=checking_fixed_KerPar)
check_v_fixed_s.set(1)
fixed_KerPar_but_check.grid(row=2, column=0, sticky=N+S+E+W)
fixed_KerPar_entry=Entry(kernel_frame, justify=CENTER)
fixed_KerPar_entry.insert(0,1)
fixed_KerPar_entry.grid(row=2, column=1, sticky=N+S+E+W)

autosigma_but_check=Checkbutton(kernel_frame, text='Autosigma', state=DISABLED, variable=check_v_auto, command=checking_auto)
check_v_auto.set(0)
autosigma_but_check.grid(row=3, column=0, sticky=N+S+E+W)

# ????? Choosing Sigma ??????
# ????? AutoSigma ??????





#Learning Frame *************************************************
learning_frame=Frame(root)

learning_title=Label(learning_frame, text='Learning', font=('Verdana', 12, 'bold'))
learning_title.grid(row=0, column=0, sticky=N+S+E+W)

# ????? Using KCV ??????

def checking_KCV():
    if check_v_KCV.get()==1:
        check_v_fixed.set(0)
        fixed_l_entry.config(state='disabled')
        Split_type_menu.config(state='active')
        n_split_entry.config(state='active')
        tmin_entry.config(state='active')
        tmax_entry.config(state='active')
        nt_values_entry.config(state='active')
        Space_type_menu.config(state='active')
        filter_type=var_filter.get()
        filter_choice(filter_type)
    elif check_v_KCV.get()==0:
        check_v_fixed.set(1)
        Split_type_menu.config(state='disabled')
        n_split_entry.config(state='disabled')
        tmin_entry.config(state='disabled')
        tmax_entry.config(state='disabled')
        nt_values_entry.config(state='disabled')
        Space_type_menu.config(state='disabled')
        fixed_l_entry.config(state='active')

def checking_fixed():
    if check_v_fixed.get()==1:
        check_v_KCV.set(0)
        Split_type_menu.config(state='disabled')
        n_split_entry.config(state='disabled')
        tmin_entry.config(state='disabled')
        tmax_entry.config(state='disabled')
        nt_values_entry.config(state='disabled')
        Space_type_menu.config(state='disabled')
        fixed_l_entry.config(state='active')
    elif check_v_fixed.get()==0:
        check_v_KCV.set(1)
        fixed_l_entry.config(state='disabled')
        Split_type_menu.config(state='active')
        n_split_entry.config(state='active')
        tmin_entry.config(state='active')
        tmax_entry.config(state='active')
        nt_values_entry.config(state='active')
        Space_type_menu.config(state='active')
        filter_type=var_filter.get()
        filter_choice(filter_type)

check_v_KCV=IntVar()
check_v_fixed=IntVar()

fixed_l_but_check=Checkbutton(learning_frame, text='Use fixed l value', variable=check_v_fixed, command=checking_fixed)
check_v_fixed.set(1)
fixed_l_but_check.grid(row=8, column=0, sticky=N+S+E+W)

fixed_l_entry=Entry(learning_frame, justify=CENTER)
fixed_l_entry.insert(0,0.01)
fixed_l_entry.grid(row=8, column=1, sticky=N+S+E+W)

KCV_but_check=Checkbutton(learning_frame, text='Use KCV', variable=check_v_KCV, command=checking_KCV)
check_v_KCV.set(0)
KCV_but_check.grid(row=1, column=0, sticky=N+S+E+W)

Split=Label(learning_frame, text='Split')
Split.grid(row=2, column=0, sticky=N+S+E+W)
var_Split=StringVar(learning_frame)
Split_type_menu=OptionMenu(learning_frame, var_Split,'Select split type', 'Sequential', 'Random')
Split_type_menu.grid(row=2, column=1)

n_split=Label(learning_frame, text='# split')
n_split.grid(row=3, column=0, sticky=N+S+E+W)
n_split_entry=Entry(learning_frame, justify=CENTER)
n_split_entry.insert(0, 5)
n_split_entry.grid(row=3, column=1)

tmin_l=Label(learning_frame, text='t min')
tmin_l.grid(row=4, column=0, sticky=N+S+E+W)
tmin_entry=Entry(learning_frame, justify=CENTER)
tmin_entry.insert(0, 0.001)
tmin_entry.grid(row=4, column=1)

tmax_l=Label(learning_frame, text='t max')
tmax_l.grid(row=5, column=0, sticky=N+S+E+W)
tmax_entry=Entry(learning_frame, justify=CENTER)
tmax_entry.insert(0, 1)
tmax_entry.grid(row=5, column=1)

nt_values_l=Label(learning_frame, text='# of t values')
nt_values_l.grid(row=6, column=0, sticky=N+S+E+W)
nt_values_entry=Entry(learning_frame, justify=CENTER)
nt_values_entry.insert(0, 10)
nt_values_entry.grid(row=6, column=1)

Space=Label(learning_frame, text='Space')
Space.grid(row=7, column=0, sticky=N+S+E+W)
var_Space=StringVar(learning_frame)
Space_type_menu=OptionMenu(learning_frame, var_Space,'Select space type', 'Linear space', 'Log space')
Space_type_menu.grid(row=7, column=1)

Split_type_menu.config(state='disabled')
n_split_entry.config(state='disabled')
tmin_entry.config(state='disabled')
tmax_entry.config(state='disabled')
nt_values_entry.config(state='disabled')
Space_type_menu.config(state='disabled')






#Results Frame ******************************************************
results_frame=Frame(root)

results_title=Label(results_frame, text='Results', font=('Verdana', 12, 'bold'))
results_title.grid(row=0, column=0, sticky=N+S+E+W)

tr_error=' - - - '
ts_error=' - - - '
select_t=' - - - '
select_sigma=' - - - '

tr_error_title=Label(results_frame, text='Training error')
tr_error_title.grid(row=1, column=0, sticky=N+S+E+W)
tr_error_v=Label(results_frame, text=tr_error)
tr_error_v.grid(row=1, column=2, sticky=N+S+E+W)

ts_error_title=Label(results_frame, text='Test error')
ts_error_title.grid(row=2, column=0, sticky=N+S+E+W)
ts_error_v=Label(results_frame, text=ts_error)
ts_error_v.grid(row=2, column=2, sticky=N+S+E+W)

select_t_title=Label(results_frame, text='Selected t')
select_t_title.grid(row=3, column=0, sticky=N+S+E+W)
select_t_v=Label(results_frame, text=select_t)
select_t_v.grid(row=3, column=2, sticky=N+S+E+W)

select_sigma_title=Label(results_frame, text='Selected sigma')
select_sigma_title.grid(row=4, column=0, sticky=N+S+E+W)
select_sigma_v=Label(results_frame, text=select_sigma)
select_sigma_v.grid(row=4, column=2, sticky=N+S+E+W)

def change_results(tr_error, ts_error, select_t, select_sigma):
    tr_error=str(tr_error)
    ts_error=str(ts_error)
    select_t=str(select_t)
    select_sigma=str(select_sigma)
    tr_error_v.config(text=tr_error)
    ts_error_v.config(text=ts_error)
    select_t_v.config(text=select_t)
    select_sigma_v.config(text=select_sigma)
    return





#Eplot_frame Frame **************************************************
eplot_frame=Frame(root)

#eplot_title=Label(eplot_frame, text='KCV - Error Plot', font=('Verdana', 12, 'bold'))
#eplot_title.grid(row=0, column=0, sticky=N+S+E+W)

error_plot=Figure(figsize=(4,4))

subplot_error_t=error_plot.add_subplot(1,1,1)
subplot_error_t.set_title('KCV - Error plot')
#subplot_error_v=error_plot.add_subplot(1,1,1)
e_empty1=0
e_empty2=0
subplot_error_t.plot(e_empty1, e_empty2)
#subplot_error_v.plot(e_empty1, e_empty2)


e_canvas=FigureCanvasTkAgg(error_plot, eplot_frame)          
e_canvas.get_tk_widget().grid(row=0, column=0)


#Plot Frame *****************************************************
plot_frame=Frame(root)

plot_title=Label(plot_frame, text='Plot', font=('Verdana', 12, 'bold'))
plot_title.grid(row=0, column=0, sticky=N+S+E+W)

main_plot=Figure(figsize=(11,4))

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

def def_filter(X1, X2):
    filter_type=var_filter.get()
    if np.array_equal(filter_type, 'Reg. Least Squared'):
        filter=SquareDist(X1, X2)
    else:
        filter= 'Reg. Least Squared is only filter'
        print filter
    return filter
 
def apply_classify():
    tkMessageBox.showinfo('Info', 'Computing...')
    
    if check_v_fixed.get()==1:
        subplot_ts.hold(False)
        subplot_tr.hold(False)
        subplot_classify_ts.hold(False)
        subplot_classify_tr.hold(False)
        
        datafile = np.load('loadeddata.npz')
        Xtr=datafile['Xtr']
        Ytr=datafile['Ytr']
        Xts=datafile['Xts']
        Yts=datafile['Yts']
        
        if check_v_fixed_s.get()==1:
            s_value=float(fixed_KerPar_entry.get())
        elif check_v_auto.get()==1:
            s_value=autosigma(Xtr, 5)
        l_value=float(fixed_l_entry.get())
        kernel_type=var_kernel.get()
        if kernel_type=='Linear':
            s_value=[]
        
        filter_type=var_filter.get()
        trange=[float(fixed_l_entry.get())]
        if check_v_class.get()==1:
            task='Classification'
        elif check_v_class.get()==0:
            task='Regression'
        
        alpha, err = learn(kernel_type, s_value, filter_type, trange, Xtr, Ytr, task)
        min_err = min(err)
        index = np.argmin(err)
        K=KernelMatrix(Xts, Xtr, kernel_type, s_value)
        y_learnt = np.dot(K, alpha[:,index])
        lrn_error = learn_error(y_learnt, Xts, task)
        
        step = 0.1
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
        y_learnt=np.dot(K, alpha[:, index])
        z=np.array(np.reshape(y_learnt,(np.size(a,axis=1),np.size(a,axis=0))).T)
        z=np.array(z)
        
        
        subplot_classify_ts.contourf(a, b, z, 1, colors=('w', 'b'), alpha=.3, antialiased=True)
        subplot_classify_tr.contourf(a, b, z, 1, colors=('w', 'b'), alpha=.3, antialiased=True)
        canvas=FigureCanvasTkAgg(main_plot, plot_frame)
        canvas.get_tk_widget().grid(row=1, column=0)
        subplot_ts.hold(True)
        subplot_tr.hold(True)
        subplot_ts.scatter(Xts[:,0],Xts[:,1],25,Yts,edgecolor='None')
        subplot_tr.scatter(Xtr[:,0],Xtr[:,1],25,Ytr,edgecolor='None')
        canvas=FigureCanvasTkAgg(main_plot, plot_frame)
        canvas.get_tk_widget().grid(row=1, column=0)
        
        
        select_t=l_value
        select_sigma=s_value
        ts_error=' - - - '
        tr_error=' - - - '
        
        change_results(tr_error, ts_error, select_t, select_sigma)
        
    elif check_v_KCV.get()==1:
        subplot_ts.hold(False)
        subplot_tr.hold(False)
        subplot_classify_ts.hold(False)
        subplot_classify_tr.hold(False)
        
        datafile = np.load('loadeddata.npz')
        Xtr=datafile['Xtr']
        Ytr=datafile['Ytr']
        Xts=datafile['Xts']
        Yts=datafile['Yts']
        
        subplot_ts.scatter(Xts[:,0],Xts[:,1],25,Yts,edgecolor='None')
        subplot_tr.scatter(Xtr[:,0],Xtr[:,1],25,Ytr,edgecolor='None')
        canvas=FigureCanvasTkAgg(main_plot, plot_frame)
        canvas.get_tk_widget().grid(row=1, column=0)
        
        if check_v_fixed_s.get()==1:
            s_value=float(fixed_KerPar_entry.get())
            #s_value=[s_value]
        elif check_v_auto.get()==1:
            s_value=float(autosigma(Xtr, 5))
            #s_value=[s_value]
            
        kernel_type=var_kernel.get()
        if kernel_type=='Linear':
            s_value=[]
            
        tmin=float(tmin_entry.get())
        tmax=float(tmax_entry.get())
        nt_values=float(nt_values_entry.get())
        space_type=var_Space.get()
        
        if check_v_class.get()==1:
            task='Classification'
        elif check_v_class.get()==1:
            task='Regression'
            
        split_type=var_Split.get()
        k=float(n_split_entry.get())
        filter_type=var_filter.get()
        if filter_type=='Reg. Least Squared' or filter_type=='Truncated SVD' or filter_type=='Spectral Cut-Off':
            trange=np.linspace(tmin, tmax, nt_values)
        elif filter_type=='Landweber' or filter_type=='NU-method':
            trange=[tmax]
        
        t_kcv_idx, avg_err_kcv=kcv(kernel_type, s_value, filter_type, trange, Xtr, Ytr, k, task, split_type)
        print t_kcv_idx
        L=trange[t_kcv_idx]
        trange=np.ravel(trange)
        #L, S, Vm, Vs, Tm, Ts = holdoutCVKernRLS(Xtr, Ytr, kernel_type, s_value, tmin, tmax, nt_values, space_type)
        
       
        print type(L), L
        subplot_ts.hold(True)
        subplot_tr.hold(True)
        
        alpha, err = learn(kernel_type, s_value, filter_type, trange, Xtr, Ytr, task)
        min_err = min(err)
        index = np.argmin(err)
        K=KernelMatrix(Xts, Xtr, kernel_type, s_value)
        y_learnt = np.dot(K, alpha[:,index])
        lrn_error = learn_error(y_learnt, Xts, task)
        
        step = 0.1
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
        y_learnt=np.dot(K, alpha[:, index])
        z=np.array(np.reshape(y_learnt,(np.size(a,axis=1),np.size(a,axis=0))).T)
        z=np.array(z)
        
        
        subplot_classify_ts.contourf(a, b, z, 1, colors=('w', 'b'), alpha=.3, antialiased=True)
        subplot_classify_tr.contourf(a, b, z, 1, colors=('w', 'b'), alpha=.3, antialiased=True)
        canvas=FigureCanvasTkAgg(main_plot, plot_frame)
        canvas.get_tk_widget().grid(row=1, column=0)
        subplot_ts.hold(True)
        subplot_tr.hold(True)
        subplot_ts.scatter(Xts[:,0],Xts[:,1],25,Yts,edgecolor='None')
        subplot_tr.scatter(Xtr[:,0],Xtr[:,1],25,Ytr,edgecolor='None')
        canvas=FigureCanvasTkAgg(main_plot, plot_frame)
        canvas.get_tk_widget().grid(row=1, column=0)
        
        avg_err_kcv=np.ravel(avg_err_kcv)
        print type(avg_err_kcv), np.shape(avg_err_kcv)
        print type(trange), np.shape(trange)
        subplot_error_t.hold(False)
        #subplot_error_v.hold(False)
        print 'trange', trange
        print 'avg_err_kcv', avg_err_kcv
        
        subplot_error_t.plot(trange, avg_err_kcv, 'b--', label='Test error')
        subplot_error_t.hold(True)
        #subplot_error_v.plot(trange, avg_err_kcv, 'g--', label='Validation error')
        #subplot_error_v.hold(True)
         
        subplot_error_t.plot(trange[t_kcv_idx], avg_err_kcv[t_kcv_idx], 'ro', linewidth=10.0, label='Test error')
        subplot_error_t.hold(True)
        #subplot_error_v.plot(trange[t_kcv_idx], avg_err_kcv[t_kcv_idx], 'ro', linewidth=10.0, label='Validation error')
        #subplot_error_v.hold(True)
        
        e_canvas=FigureCanvasTkAgg(error_plot, eplot_frame)          
        e_canvas.get_tk_widget().grid(row=0, column=0)
#         
        #tr_error=float(Tm[np.argmin(Vm)])
        tr_error=' - - - '
        #ts_error=float(min(Vm))
        ts_error=' - - - '
        select_t=float(L)
        select_sigma=float(s_value)
        
        change_results(tr_error, ts_error, select_t, select_sigma)
        
    return
        

run_but=Button(bottom_frame, text='Run', command=apply_classify)
run_but.grid(row=0, column=0)

# Placement ****************************************************
for root_cols in range(3):
  Grid.columnconfigure(root, root_cols, weight=1)
for root_rows in range(6):
  Grid.rowconfigure(root, root_rows, weight=1)


task_frame.grid(row=0, column=0, padx=0, pady=0, sticky=N+S+E+W)
data_frame.grid(row=1, column=0,padx=0, pady=0, sticky=N+S+E+W)
filter_frame.grid(row=2, column=0,padx=0, pady=0, sticky=N+S+E+W)
kernel_frame.grid(row=3, column=0,padx=0, pady=0, sticky=N+S+E+W)
learning_frame.grid(rowspan=2, row=0, column=1,padx=0, pady=0, sticky=N+S+E+W)
results_frame.grid(rowspan=2, row=2, column=1, padx=0, pady=0, sticky=N+S+E+W)
eplot_frame.grid(rowspan=4, row=0, column=2, padx=0, pady=0, sticky=N+S+E+W)
plot_frame.grid(row=5, columnspan=3,padx=0, pady=0, sticky=N+S+E+W)
bottom_frame.grid(row=4, columnspan=3,padx=0, pady=0, sticky=N+S+E+W)

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
for plot_cols in range(10):
  Grid.columnconfigure(plot_frame, plot_cols, weight=1)
for plot_rows in range(10):
  Grid.rowconfigure(plot_frame, plot_rows, weight=1)
  
#Bottom Frame
for bottom_cols in range(10):
  Grid.columnconfigure(bottom_frame, bottom_cols, weight=1)
for bottom_rows in range(10):
  Grid.rowconfigure(bottom_frame, bottom_rows, weight=1)
  
# *****************************************************************


root.mainloop()
