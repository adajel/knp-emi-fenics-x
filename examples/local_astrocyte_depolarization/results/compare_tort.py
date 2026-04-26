import numpy as np
import sys

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

#blue = '#3d83c4'
blue = "#0000FF"
grey = "#423c3c"
green = "#057a69"
pink = '#e31be3'

blue_light = "#56B4E9"
dark_blue = "#1C3A5A"
pink_light = "#EC407A"
dark_pink = "#AD1457"
orange = "#FF9D3A"

# set font & text parameters
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'

def read_me(fname):
    with open(fname) as f:
        lines = f.readlines()
        x = [float(line.split()[0]) for line in lines]
        return x
        #return x[:240]

fdirs = "ICS-tort-x5"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_I = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_I = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_I = read_me(fname)
fname = f"{fdirs}/i_kir_glial.txt"
I_Kir_I = read_me(fname)

#fname = f"{fdirs}/I_Na_glial.txt"
#I_Na_I = read_me(fname)
#fname = f"{fdirs}/I_Cl_glial.txt"
#I_Cl_I = read_me(fname)
#fname = f"{fdirs}/I_pump_glial.txt"
#I_pump_I = read_me(fname)
fname = f"{fdirs}/E_Cl_glial.txt"
E_Cl_I = read_me(fname)
fname = f"{fdirs}/E_Na_glial.txt"
E_Na_I = read_me(fname)
fname = f"{fdirs}/E_K_glial.txt"
E_K_I = read_me(fname)

fdirs = "ECS-tort-x5"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_E = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_E = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_E = read_me(fname)
fname = f"{fdirs}/i_kir_glial.txt"
I_Kir_E = read_me(fname)



#fname = f"{fdirs}/I_Na_glial.txt"
#I_Na_E = read_me(fname)
#fname = f"{fdirs}/I_Cl_glial.txt"
#I_Cl_E = read_me(fname)
#fname = f"{fdirs}/I_pump_glial.txt"
#I_pump_E = read_me(fname)
fname = f"{fdirs}/E_Cl_glial.txt"
E_Cl_E = read_me(fname)
fname = f"{fdirs}/E_Na_glial.txt"
E_Na_E = read_me(fname)
fname = f"{fdirs}/E_K_glial.txt"
E_K_E = read_me(fname)

# get phi_M time
fdirs = "baseline"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_bs = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_bs = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_bs = read_me(fname)
fname = f"{fdirs}/E_Cl_glial.txt"
E_Cl_bs = read_me(fname)
fname = f"{fdirs}/E_Na_glial.txt"
E_Na_bs = read_me(fname)
fname = f"{fdirs}/E_K_glial.txt"
E_K_bs = read_me(fname)
fname = f"{fdirs}/i_pump_glial.txt"
I_pump_bs = read_me(fname)
fname = f"{fdirs}/i_kir_glial.txt"
I_Kir_bs = read_me(fname)

# time
dt = 0.1
save_frequency = 5
Tstop = 300
t = np.arange(0, Tstop, dt * save_frequency)

# get index of max value (i.e. where the stimuli is turned off) - same for all
# model variations
stimuli_end = np.argmax(phi_M_I) + 25
print(f"stimuli end: {stimuli_end*0.1*5}")

def get_normalized_phi_M(phi_M):

    # Normalized membrane potential over time
    phi_M_max = phi_M[stimuli_end]  # get max value of membrane potential
    phi_M_rest = np.min(phi_M)      # get min value of membrane potential (i.e. the resting potential)

    # calculate normalized membrane potential
    N = len(phi_M)
    phi_M_norm = (phi_M[stimuli_end:] - np.full(N, phi_M_rest)[stimuli_end:])/(phi_M_max - phi_M_rest)

    return phi_M_norm

def get_normalized_phi_M_space(phi_M):

    # Normalized membrane potential over time
    phi_M_max = np.max(phi_M)       # get max value of membrane potential
    phi_M_rest = np.min(phi_M)      # get min value of membrane potential (i.e. the resting potential)

    # calculate normalized membrane potential
    N = len(phi_M)
    phi_M_norm = (phi_M[int(len(phi_M)/2):] - np.full(N, \
        phi_M_rest)[int(len(phi_M)/2):])/(phi_M_max - phi_M_rest)

    return phi_M_norm


### ------------------------------------------------------------ ###
### Make plot concentrations, potential and normalized potential ###
### ------------------------------------------------------------ ###

lw = 4

fig = plt.figure(figsize=(20, 15))
ax = plt.gca()

phi_M_norm_bs = get_normalized_phi_M(phi_M_bs)
phi_M_norm_I = get_normalized_phi_M(phi_M_I)
phi_M_norm_E = get_normalized_phi_M(phi_M_E)

#print(phi_M_norm_E)
#print(phi_M_norm_I)
#print(phi_M_norm_bs)

#exit(0)

indices_bs = [i for i, x in enumerate(phi_M_norm_bs) if (x > 0.49 and x < 0.51)]
indices_I = [i for i, x in enumerate(phi_M_norm_I) if (x > 0.49 and x < 0.51)]
indices_E = [i for i, x in enumerate(phi_M_norm_E) if (x > 0.49 and x < 0.51)]

print(indices_I)
print(indices_E)
print(indices_bs)
print("time constant bs", indices_bs[0]*dt*save_frequency)
print("time constant I", indices_I[0]*dt*save_frequency)
print("time constant E", indices_E[0]*dt*save_frequency)

t_normalized = np.arange(stimuli_end * dt * save_frequency, Tstop, dt * save_frequency)

#phi_M_space_norm = get_normalized_phi_M_space(phi_M_space)
#t_normalized_space = np.linspace(100, 200, int(3200/2))
#x = np.linspace(0, 200, 3200)

ax1 = fig.add_subplot(3,4,1)
plt.plot(t, K_ECS_E, linewidth=lw, color=pink)
plt.plot(t, K_ECS_I, linewidth=lw, color=blue)
plt.plot(t, K_ECS_bs, linewidth=lw, color=green, linestyle="dotted")
plt.ylabel(r"$c_{K_e}$ (mM)")
plt.xlabel(r"time (ms)")
#plt.xticks([0, 60, 120])

ax1 = fig.add_subplot(3,4,2)
plt.plot(t, np.array(E_K_E), linewidth=lw, color=pink)
plt.plot(t, np.array(E_K_I), linewidth=lw, color=blue)
plt.plot(t, np.array(E_K_bs), linewidth=lw, color=green, linestyle="dotted")
plt.ylabel(r"$\rm E_{K}$ (mV)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(3,4,5)
plt.ylabel(r"$I$ ($\rm \mu A/cm^2$)")
plt.plot(t, np.array(I_Kir_E), linewidth=lw, color=pink)
plt.plot(t, np.array(I_Kir_I), linewidth=lw, color=blue)
plt.plot(t, np.array(I_Kir_bs), linewidth=lw, color=green, linestyle="dotted")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(3,4,6)
plt.plot(t, K_ICS_E, linewidth=lw, color=pink)
plt.plot(t, K_ICS_I, linewidth=lw, color=blue)
plt.plot(t, K_ICS_bs, linewidth=lw, color=green, linestyle="dotted")
plt.ylabel(r"$c_{K_g}$ (mM)")
plt.xlabel(r"time (ms)")
#plt.xticks([0, 60, 120])

ax1 = fig.add_subplot(3,4,9)
plt.plot(t, phi_M_E, linewidth=lw, color=pink, label=r"E")
plt.plot(t, phi_M_I, linewidth=lw, color=blue, label=r"I")
plt.plot(t, phi_M_bs, linewidth=lw, color=green, linestyle="dotted", label=r"bs")
#plt.axvline(x=112.5, color='red', linestyle='--', label='Line at $x=6$')
plt.axvline(x=104.5, color='red', linestyle='--', label='Line at $x=6$')
plt.ylabel(r"$\phi_M$ (mV)")
plt.xlabel(r"time (ms)")
#plt.xticks([0, 60, 120])

ax1 = fig.add_subplot(3,4,10, xlim=[112, 305])
plt.plot(t_normalized, phi_M_norm_E, linewidth=lw, color=pink, label=r"ECS $\lambda \times 5$")
plt.plot(t_normalized, phi_M_norm_I, linewidth=lw, color=blue, label=r'ICS $\lambda \times 5$')
plt.plot(t_normalized, phi_M_norm_bs, linewidth=lw, color=green, linestyle="dotted", label=r'baseline')
plt.plot([stimuli_end * dt * save_frequency, Tstop], [0.5, 0.5], color='grey', linestyle="dotted", linewidth=lw)
plt.ylabel(r"normalized $\phi_M$")
plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.xticks([100, 150, 200, 250, 300])
plt.xlabel(r"time (ms)")
plt.legend()

# make pretty
ax.axis('off')
plt.tight_layout()

# save figure to file
plt.savefig(f'tort.svg', format='svg')
plt.savefig(f'tort.png', format='png')
