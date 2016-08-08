import numpy as np

from Tkinter import *
from ttk import *
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from dataset_scripts import *

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