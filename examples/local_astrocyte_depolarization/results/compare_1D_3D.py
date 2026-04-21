import numpy as np
import sys

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

blue = '#3d83c4'
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

fdirs = "1D"
fname = f"{fdirs}/phi_M_no_dec.txt"
phi_M_1D = read_me(fname)
fname = f"{fdirs}/K_ECS_no_dec.txt"
K_ECS_1D = read_me(fname)
fname = f"{fdirs}/K_ICS_no_dec.txt"
K_ICS_1D = read_me(fname)
fname = f"{fdirs}/phi_M_space.txt"
phi_M_space = read_me(fname)

fname = f"{fdirs}/I_Kir.txt"
I_Kir_1D = read_me(fname)
fname = f"{fdirs}/I_Na.txt"
I_Na_1D = read_me(fname)
fname = f"{fdirs}/I_Cl.txt"
I_Cl_1D = read_me(fname)
fname = f"{fdirs}/I_pump.txt"
I_pump_1D = read_me(fname)

fname = f"{fdirs}/E_Cl.txt"
E_Cl_1D = read_me(fname)
fname = f"{fdirs}/E_Na.txt"
E_Na_1D = read_me(fname)
fname = f"{fdirs}/E_K.txt"
E_K_1D = read_me(fname)


# get phi_M time
fdirs = "baseline"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_3D = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_3D = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_3D = read_me(fname)

# time
dt = 0.1
save_frequency = 1
Tstop = 105
t = np.arange(0, Tstop, dt * save_frequency)

# get index of max value (i.e. where the stimuli is turned off) - same for all
# model variations
stimuli_end = np.argmax(phi_M_1D)

def get_normalized_phi_M(phi_M):

    # Normalized membrane potential over time
    phi_M_max = np.max(phi_M)       # get max value of membrane potential
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

phi_M_norm_1D = get_normalized_phi_M(phi_M_1D)
phi_M_norm_3D = get_normalized_phi_M(phi_M_3D)

t_normalized = np.arange(stimuli_end * dt * save_frequency, Tstop, dt * save_frequency)

phi_M_space_norm = get_normalized_phi_M_space(phi_M_space)
t_normalized_space = np.linspace(100, 200, int(3200/2))
x = np.linspace(0, 200, 3200)

ax1 = fig.add_subplot(3,4,1)
plt.plot(t, K_ECS_1D, linewidth=lw, color=pink, label=r'1D')
#plt.plot(t, K_ECS_3D, linewidth=lw, color=blue, label=r"3D")
plt.ylabel(r"$c_{K_e}$ (mM)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(3,4,2)
plt.plot(t, K_ICS_1D, linewidth=lw, color=pink, label=r'1D')
#plt.plot(t, K_ICS_3D, linewidth=lw, color=blue, label=r"3D")
plt.ylabel(r"$c_{K_g}$ (mM)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(3,4,3)
plt.plot(t, np.array(phi_M_1D)*1.0e3, linewidth=lw, color=pink, label=r'1D')
#plt.plot(t, phi_M_3D, linewidth=lw, color=blue, label=r"3D")
plt.ylabel(r"$\phi_M$ (mV)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(3,4,4)
plt.plot(t_normalized, phi_M_norm_1D, linewidth=lw, color=pink, label=r'1D')
#plt.plot(t_normalized, phi_M_norm_3D, linewidth=lw, color=blue, label=r"3D")
plt.plot([stimuli_end * dt * save_frequency, Tstop], [0.5, 0.5], color='grey', linestyle="dotted", linewidth=lw)
plt.ylabel(r"normalized $\phi_M$")
plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.xlabel(r"time (ms)")
plt.legend()

ax1 = fig.add_subplot(3,4,7)
plt.plot(np.array(E_K_1D), linewidth=lw, color=blue_light, label=r'1D')
plt.ylabel(r"$\rm E_{K}$ (mV)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(3,4,8)
plt.ylabel(r"$I$ ()")
plt.plot(np.array(I_Kir_1D), linewidth=lw, color=blue, label=r'$I_{\rm Kir}$')
plt.plot(- 2 * np.array(I_pump_1D), linewidth=lw, color=dark_blue, label=r'$I^{\rm K}_{\rm pump}$')
plt.plot(np.array(I_Na_1D), linewidth=lw, color=pink_light, label=r'$I_{\rm Na}$')
plt.plot(3 * np.array(I_pump_1D), linewidth=lw, color=dark_pink, label=r'$I^{\rm Na}_{\rm pump}$')
plt.plot(np.array(I_Cl_1D), linewidth=lw, color=orange, label=r'$I_{\rm Cl}$')
plt.xlabel(r"time (ms)")
plt.legend()

ax1 = fig.add_subplot(3,4,11)
plt.plot(x, np.array(phi_M_space)*1.0e3, linewidth=lw, color=green, label=r'1D')
plt.ylabel(r"$\phi_M$ (mV)")
plt.xticks([0, 50, 100, 150, 200])
plt.xlabel(r"$x(\mu\rm{m})$")

ax1 = fig.add_subplot(3,4,12)
plt.plot(t_normalized_space, phi_M_space_norm, linewidth=lw, color=green, label=r'1D')
plt.plot([100, 200], [0.36787944117, 0.36787944117], color='grey', linestyle="dotted", linewidth=lw)
plt.ylabel(r"normalized $\phi_M$")
plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.xticks([100, 125, 150, 175, 200])
plt.xlabel(r"$x(\mu\rm{m})$")

# make pretty
ax.axis('off')
plt.tight_layout()

# save figure to file
plt.savefig(f'1D-3D.svg', format='svg')
plt.savefig(f'1D-3D.png', format='png')
