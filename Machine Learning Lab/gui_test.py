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

print range(4)
kernel_frame=Frame(root)
Grid.rowconfigure(root, 0, weight=1)
Grid.columnconfigure(root, 0, weight=1)
kernel_frame.grid(row=0, column=0, padx=0, pady=0, sticky=N+S+E+W)

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

for x in range(2):
  Grid.columnconfigure(kernel_frame, x, weight=1)

for y in range(4):
  Grid.rowconfigure(kernel_frame, y, weight=1)

root.mainloop()