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
from ttk import *

#from dataGeneration import create_data
from dataset_scripts import gaussian
from dataset_scripts import linear_data
from dataset_scripts import moons
from dataset_scripts import sinusoidal
from dataset_scripts import spiral

from spectral_reg_toolbox import EuclideanDistance
from spectral_reg_toolbox import autosigma
from spectral_reg_toolbox import holdoutCVKernRLS
from spectral_reg_toolbox import KernelMatrix
from spectral_reg_toolbox import regularizedKernLSTest
from spectral_reg_toolbox import regularizedKernLSTrain
from spectral_reg_toolbox import separatingFKernRLS


root=Tk()

root.iconbitmap('simula.ico')
root.wm_title('Machine Learning Lab')

s=Style()
s.theme_use('vista')

# Input Frame ****************************************************
#input_frame=Frame(root)

#input_title=Label(input_frame, text='Input', font=('Verdana', 12, 'bold'))
#input_title.grid(row=0, column=0, sticky=W)


# Task Frame *****************************************************
task_frame=Frame(root)
#task_frame.grid(row=1, column=0, sticky=W)

task_title=Label(task_frame, text='Task', font=('Verdana', 12, 'bold'))
task_title.grid(row=0, column=0, sticky=W)

def keep_task_check():
    check_v_class.set(1)

check_v_class=IntVar()
class_but_check=Checkbutton(task_frame, text='Classification', variable=check_v_class, command=keep_task_check)
check_v_class.set(1)
class_but_check.grid(row=1, column=0, sticky=W)

check_v_reg=IntVar()
reg_but_check=Checkbutton(task_frame, text='Regression', variable=check_v_reg, state=DISABLED)
check_v_reg.set(0)
reg_but_check.grid(row=1, column=1, sticky=E)




# Data Frame *****************************************************
data_frame=Frame(root)
#data_frame.grid(row=2, column=0, sticky=W)

# ??? Existing dataset ???
data_title=Label(data_frame, text='Data type', font=('Verdana', 12, 'bold'))
data_title.grid(row=0, column=0, sticky=W)

var_data=StringVar(data_frame)
data_type_menu=OptionMenu(data_frame, var_data,'Select data type', 'Gaussian', 'Moons', 'Spiral', 'Sinusoidal', 'Linear')
data_type_menu.grid(row=1, column=1)

ntr_samples=Label(data_frame, text='# training samples')
ntr_samples.grid(row=2, column=0, sticky=E)
ntr_samples_entry=Entry(data_frame, justify=CENTER)
ntr_samples_entry.insert(0,100)
ntr_samples_entry.grid(row=2, column=1)

nts_samples=Label(data_frame, text='# test samples')
nts_samples.grid(row=3, column=0, sticky=E)
nts_samples_entry=Entry(data_frame, justify=CENTER)
nts_samples_entry.insert(0, 1000)
nts_samples_entry.grid(row=3, column=1)

flip_ratio=Label(data_frame, text='Wrong label ratio')
flip_ratio.grid(row=4, column=0, sticky=E)
flip_ratio_entry=Entry(data_frame, justify=CENTER)
flip_ratio_entry.insert(0, 0.01)
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
        Xtr, Ytr = gaussian(gaussian_input_tr, pflip)
        Xts, Yts = gaussian(gaussian_input_ts, pflip)
    elif data_type=='Moons':
        moons_input_tr=[ntr1, ntr2]
        moons_input_ts=[nts1, nts2]
        Xtr, Ytr = moons(moons_input_tr, pflip)
        Xts, Yts = moons(moons_input_ts, pflip)
    elif data_type=='Spiral':
        spiral_input_tr=[ntr1, ntr2]
        spiral_input_ts=[nts1, nts2]
        Xtr, Ytr = spiral(spiral_input_tr, pflip)
        Xts, Yts = spiral(spiral_input_ts, pflip)
    elif data_type=='Sinusoidal':
        sinusoidal_input_tr=[ntr1, ntr2]
        sinusoidal_input_ts=[nts1, nts2]
        Xtr, Ytr = sinusoidal(sinusoidal_input_tr, pflip)
        Xts, Yts = sinusoidal(sinusoidal_input_ts, pflip)
    elif data_type=='Linear':
        linear_data_input_tr=[ntr1, ntr2]
        linear_data_input_ts=[nts1, nts2]
        Xtr, Ytr = linear_data(linear_data_input_tr, pflip)
        Xts, Yts = linear_data(linear_data_input_ts, pflip)
    np.savez('loadeddata',Xtr=Xtr,Ytr=Ytr,Xts=Xts,Yts=Yts)
      
    subplot_ts.hold(False)
    subplot_tr.hold(False)
    subplot_ts.scatter(Xts[:,0],Xts[:,1],50,Yts,edgecolor='None')
    subplot_tr.scatter(Xtr[:,0],Xtr[:,1],50,Ytr,edgecolor='None')
    canvas=FigureCanvasTkAgg(main_plot, plot_frame)
    canvas.get_tk_widget().grid(row=1, column=0) 
      
    return Xtr, Ytr, Xts, Yts

load_data_but=Button(data_frame, text='Load data', command=create_data)
load_data_but.grid(row=5, column=1, sticky=E)






#Filter Frame ****************************************************
filter_frame=Frame(root)

filter_title=Label(filter_frame, text='Filter', font=('Verdana', 12, 'bold'))
filter_title.grid(row=0, column=0, sticky=W)


var_filter=StringVar(filter_frame)
var_filter.set('Reg. Least Squared')
filter_type_menu=OptionMenu(filter_frame, var_filter, 'Reg. Least Squared')
filter_type_menu.grid(row=1, column=0, sticky=W)




    
#Kernel Frame ****************************************************
kernel_frame=Frame(root)

kernel_title=Label(kernel_frame, text='Kernel', font=('Verdana', 12, 'bold'))
kernel_title.grid(row=0, column=0, sticky=W)

def kernel_choice(kernel):
    kernel=var_kernel.get()
    if kernel=='Linear':
        fixed_KerPar_entry.config(state='disabled')
        fixed_KerPar_but_check.config(state='disabled')
        autosigma_but_check.config(state='disabled')
    elif kernel=='Polynomial':
        fixed_KerPar_entry.config(state='active')
        fixed_KerPar_but_check.config(state='active')
        autosigma_but_check.config(state='disabled')
        if check_v_fixed_s.get()==0:
            check_v_fixed_s.set(1)
    elif kernel=='Gaussian':
        fixed_KerPar_entry.config(state='active')
        fixed_KerPar_but_check.config(state='active')
        autosigma_but_check.config(state='active')

def checking_auto():
    check_v_fixed_s.set(0)
    fixed_KerPar_entry.config(state='disabled')
    
def checking_fixed_KerPar():
    check_v_auto.set(0)
    fixed_KerPar_entry.config(state='active')           

check_v_fixed_s=IntVar()
check_v_auto=IntVar()

var_kernel=StringVar(kernel_frame)
kernel_type_menu=OptionMenu(kernel_frame, var_kernel,'Select kernel', 'Linear', 'Polynomial', 'Gaussian', command=kernel_choice)
kernel_type_menu.grid(row=1, column=0, sticky=W)

fixed_KerPar_but_check=Checkbutton(kernel_frame, text='Use fixed s value', variable=check_v_fixed_s, command=checking_fixed_KerPar)
check_v_fixed_s.set(1)
fixed_KerPar_but_check.grid(row=2, column=0, sticky=E)
fixed_KerPar_entry=Entry(kernel_frame, justify=CENTER)
fixed_KerPar_entry.insert(0,1)
fixed_KerPar_entry.grid(row=2, column=1, sticky=E)

autosigma_but_check=Checkbutton(kernel_frame, text='Autosigma', state=DISABLED, variable=check_v_auto, command=checking_auto)
check_v_auto.set(0)
autosigma_but_check.grid(row=3, column=0, sticky=W)

# ????? Choosing Sigma ??????
# ????? AutoSigma ??????





#Learning Frame *************************************************
learning_frame=Frame(root)

learning_title=Label(learning_frame, text='Learning', font=('Verdana', 12, 'bold'))
learning_title.grid(row=0, column=0, sticky=W)

# ????? Using KCV ??????

def checking_KCV():
    check_v_fixed.set(0)
    fixed_l_entry.config(state='disabled')
    #Split_type_menu.config(state='active')
    #n_split_entry.config(state='active')
    tmin_entry.config(state='active')
    tmax_entry.config(state='active')
    nt_values_entry.config(state='active')
    Space_type_menu.config(state='active') 

def checking_fixed():
    check_v_KCV.set(0)
    #Split_type_menu.config(state='disabled')
    #n_split_entry.config(state='disabled')
    tmin_entry.config(state='disabled')
    tmax_entry.config(state='disabled')
    nt_values_entry.config(state='disabled')
    Space_type_menu.config(state='disabled')
    fixed_l_entry.config(state='active')

check_v_KCV=IntVar()
check_v_fixed=IntVar()

fixed_l_but_check=Checkbutton(learning_frame, text='Use fixed l value', variable=check_v_fixed, command=checking_fixed)
check_v_fixed.set(1)
fixed_l_but_check.grid(row=8, column=0, sticky=E)

fixed_l_entry=Entry(learning_frame, justify=CENTER)
fixed_l_entry.insert(0,0.01)
fixed_l_entry.grid(row=8, column=1, sticky=E)

KCV_but_check=Checkbutton(learning_frame, text='Use KCV', variable=check_v_KCV, command=checking_KCV)
check_v_KCV.set(0)
KCV_but_check.grid(row=1, column=0, sticky=W)

Split=Label(learning_frame, text='Split')
Split.grid(row=2, column=0, sticky=E)
var_Split=StringVar(learning_frame)
Split_type_menu=OptionMenu(learning_frame, var_Split,'Select split type', 'Sequential', 'Random')
Split_type_menu.grid(row=2, column=1)

n_split=Label(learning_frame, text='# split')
n_split.grid(row=3, column=0, sticky=E)
n_split_entry=Entry(learning_frame, justify=CENTER)
n_split_entry.insert(0, 5)
n_split_entry.grid(row=3, column=1)

tmin_l=Label(learning_frame, text='t min')
tmin_l.grid(row=4, column=0, sticky=E)
tmin_entry=Entry(learning_frame, justify=CENTER)
tmin_entry.insert(0, 0.001)
tmin_entry.grid(row=4, column=1)

tmax_l=Label(learning_frame, text='t max')
tmax_l.grid(row=5, column=0, sticky=E)
tmax_entry=Entry(learning_frame, justify=CENTER)
tmax_entry.insert(0, 1)
tmax_entry.grid(row=5, column=1)

nt_values_l=Label(learning_frame, text='# of t values')
nt_values_l.grid(row=6, column=0, sticky=E)
nt_values_entry=Entry(learning_frame, justify=CENTER)
nt_values_entry.insert(0, 10)
nt_values_entry.grid(row=6, column=1)

Space=Label(learning_frame, text='Space')
Space.grid(row=7, column=0, sticky=E)
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
results_title.grid(row=0, column=0, sticky=W)

tr_error=' - - - '
ts_error=' - - - '
select_t=' - - - '
select_sigma=' - - - '

tr_error_title=Label(results_frame, text='Training error')
tr_error_title.grid(row=1, column=0, sticky=W)
tr_error_v=Label(results_frame, text=tr_error)
tr_error_v.grid(row=1, column=2, sticky=W)

ts_error_title=Label(results_frame, text='Test error')
ts_error_title.grid(row=2, column=0, sticky=W)
ts_error_v=Label(results_frame, text=ts_error)
ts_error_v.grid(row=2, column=2, sticky=W)

select_t_title=Label(results_frame, text='Selected t')
select_t_title.grid(row=3, column=0, sticky=W)
select_t_v=Label(results_frame, text=select_t)
select_t_v.grid(row=3, column=2, sticky=W)

select_sigma_title=Label(results_frame, text='Selected sigma')
select_sigma_title.grid(row=4, column=0, sticky=W)
select_sigma_v=Label(results_frame, text=select_sigma)
select_sigma_v.grid(row=4, column=2, sticky=W)

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

eplot_title=Label(eplot_frame, text='KCV - Error Plot', font=('Verdana', 12, 'bold'))
eplot_title.grid(row=0, column=0, sticky=W)

error_plot=Figure(figsize=(4,4))

subplot_error_t=error_plot.add_subplot(1,1,1)
subplot_error_v=error_plot.add_subplot(1,1,1)
e_empty1=0
e_empty2=0
subplot_error_t.plot(e_empty1, e_empty2)
subplot_error_v.plot(e_empty1, e_empty2)


e_canvas=FigureCanvasTkAgg(error_plot, eplot_frame)          
e_canvas.get_tk_widget().grid(row=1, column=0)


#Plot Frame *****************************************************
plot_frame=Frame(root)

plot_title=Label(plot_frame, text='Plot', font=('Verdana', 12, 'bold'))
plot_title.grid(row=0, column=0, sticky=W)

main_plot=Figure(figsize=(11,4))

subplot_ts=main_plot.add_subplot(1,2,1)
subplot_tr=main_plot.add_subplot(1,2,2)
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
        
        subplot_ts.scatter(Xts[:,0],Xts[:,1],50,Yts,edgecolor='None')
        subplot_tr.scatter(Xtr[:,0],Xtr[:,1],50,Ytr,edgecolor='None')
        canvas=FigureCanvasTkAgg(main_plot, plot_frame)
        canvas.get_tk_widget().grid(row=1, column=0) 
        
        if check_v_fixed_s.get()==1:
            s_value=float(fixed_KerPar_entry.get())
        elif check_v_auto.get()==1:
            s_value=autosigma(Xtr, 5)
        l_value=float(fixed_l_entry.get())
        kernel_type=var_kernel.get()
        
        subplot_ts.hold(True)
        subplot_tr.hold(True)
        
        c=regularizedKernLSTrain(Xtr, Ytr, kernel_type, s_value, l_value)
        contour_var1, contour_var2, contour_var3 = separatingFKernRLS(c, Xtr, kernel_type, s_value, Xts)
        
        subplot_classify_ts.contour(contour_var1, contour_var2, contour_var3, 1)
        subplot_classify_tr.contour(contour_var1, contour_var2, contour_var3, 1)
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
        
        subplot_ts.scatter(Xts[:,0],Xts[:,1],50,Yts,edgecolor='None')
        subplot_tr.scatter(Xtr[:,0],Xtr[:,1],50,Ytr,edgecolor='None')
        canvas=FigureCanvasTkAgg(main_plot, plot_frame)
        canvas.get_tk_widget().grid(row=1, column=0)
        
        if check_v_fixed_s.get()==1:
            s_value=float(fixed_KerPar_entry.get())
            s_value=[s_value]
        elif check_v_auto.get()==1:
            s_value=autosigma(Xtr, 5)
            s_value=[s_value]
            
        kernel_type=var_kernel.get()
        tmin=float(tmin_entry.get())
        tmax=float(tmax_entry.get())
        nt_values=float(nt_values_entry.get())
        space_type=var_Space.get()
        L, S, Vm, Vs, Tm, Ts = holdoutCVKernRLS(Xtr, Ytr, kernel_type, s_value, tmin, tmax, nt_values, space_type)
        
        subplot_ts.hold(True)
        subplot_tr.hold(True)
        
        c=regularizedKernLSTrain(Xtr, Ytr, kernel_type, S, L)
        contour_var1, contour_var2, contour_var3 = separatingFKernRLS(c, Xtr, kernel_type, S, Xts)
        
        subplot_classify_ts.contour(contour_var1, contour_var2, contour_var3, 1)
        subplot_classify_tr.contour(contour_var1, contour_var2, contour_var3, 1)
        canvas=FigureCanvasTkAgg(main_plot, plot_frame)
        canvas.get_tk_widget().grid(row=1, column=0)
        separatingFKernRLS(c, Xtr, kernel_type, S, Xts)
        
        intLambda=np.linspace(tmin, tmax, nt_values)
        intLambda=np.ndarray.tolist(intLambda)
        
        subplot_error_t.hold(False)
        subplot_error_v.hold(False)
        
        subplot_error_t.plot(intLambda, Tm, 'b--', label='Test error')
        subplot_error_t.hold(True)
        subplot_error_v.plot(intLambda, Vm, 'g--', label='Validation error')
        subplot_error_v.hold(True)
        e_canvas=FigureCanvasTkAgg(error_plot, eplot_frame)          
        e_canvas.get_tk_widget().grid(row=1, column=0)
        
        l_list=np.linspace(tmin, tmax, nt_values)
        subplot_error_t.plot(l_list[np.argmin(Tm)], min(Tm), 'ro', linewidth=10.0, label='Test error')
        subplot_error_t.hold(True)
        subplot_error_v.plot(l_list[np.argmin(Vm)], min(Vm), 'ro', linewidth=10.0, label='Validation error')
        subplot_error_v.hold(True)
        e_canvas=FigureCanvasTkAgg(error_plot, eplot_frame)          
        e_canvas.get_tk_widget().grid(row=1, column=0)
        
        tr_error=float(Tm[np.argmin(Vm)])
        ts_error=float(min(Vm))
        select_t=float(l_list[np.argmin(Vm)])
        select_sigma=S
        
        change_results(tr_error, ts_error, select_t, select_sigma)
        
    return
        

run_but=Button(bottom_frame, text='Run', command=apply_classify)
run_but.pack(side=RIGHT)

# Placement ****************************************************

task_frame.grid(row=0, column=0, padx=0, pady=0, sticky=W+N)
data_frame.grid(row=1, column=0,padx=0, pady=0, sticky=W+N)
filter_frame.grid(row=2, column=0,padx=0, pady=0, sticky=W+N)
kernel_frame.grid(row=3, column=0,padx=0, pady=0, sticky=W+N)
learning_frame.grid(rowspan=2, row=0, column=1,padx=0, pady=0, sticky=W+N)
results_frame.grid(rowspan=2, row=2, column=1, padx=0, pady=0, sticky=W+N)
eplot_frame.grid(rowspan=4, row=0, column=2, padx=0, pady=0, sticky=W+N)
plot_frame.grid(row=5, columnspan=3,padx=0, pady=0, sticky=W+N)
bottom_frame.grid(row=4, columnspan=3,padx=0, pady=0, sticky=W+N)

# *****************************************************************


root.mainloop()
