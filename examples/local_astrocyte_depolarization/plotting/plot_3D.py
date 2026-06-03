import numpy as np
import sys

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

grey = "#423c3c"
pink = '#e31be3'
blue_light = "#56B4E9"
blue_dark = "#191970"
blue = "#3975db"

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
        return np.array(x)

# get phi_M time
fdirs = "../results/baseline"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS = read_me(fname)
fname = f"{fdirs}/E_Cl_glial.txt"
E_Cl = read_me(fname)
fname = f"{fdirs}/E_Na_glial.txt"
E_Na = read_me(fname)
fname = f"{fdirs}/E_K_glial.txt"
E_K = read_me(fname)
fname = f"{fdirs}/i_kir_glial.txt"
I_Kir = read_me(fname)
fname = f"{fdirs}/i_pump_glial.txt"
I_pump = read_me(fname)
fname = f"{fdirs}/g_tot_glial.txt"
g_tot = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e = read_me(fname)

# time
dt = 0.1
save_frequency = 5
Tstop = 300
t = np.arange(0, Tstop, dt * save_frequency)

# get index of max value (i.e. where the stimuli is turned off) - same for all
# model variations
stimuli_end = np.argmax(phi_M) + 20
print(f"stimuli end: {stimuli_end*0.1*5}")

def get_normalized_phi_M(phi_M):

    # Normalized membrane potential over time
    phi_M_max = phi_M[stimuli_end]  # get max value of membrane potential
    phi_M_rest = np.min(phi_M)      # get min value of membrane potential (i.e. the resting potential)

    # calculate normalized membrane potential
    N = len(phi_M)
    phi_M_norm = (phi_M[stimuli_end:] - np.full(N, phi_M_rest)[stimuli_end:])/(phi_M_max - phi_M_rest)

    return phi_M_norm

### ------------------------------------------------------------ ###
### Make plot concentrations, potential and normalized potential ###
### ------------------------------------------------------------ ###

alpha_i = 0.11
alpha_e = 0.22
gamma_m = 4.33e4 # 1/cm

ri  = 1 / (sigma_i * alpha_i)   # intracellular resistance k Ohm cm
re  = 1 / (sigma_e * alpha_e)   # extracellular resistance k Ohm cm
rm = 1 / (g_tot * gamma_m)      # membrane resistance k Ohm cm**3
length_constant = np.sqrt(rm / (ri + re)) * 1.0e4 # convert to cm

lw = 4

fig = plt.figure(figsize=(5, 5))
ax = plt.gca()

ax1 = fig.add_subplot(1,1,1)
plt.plot(t, K_ECS, linewidth=lw, color=blue_dark)
plt.ylabel(r"$\rm c_{K_e}$ (mM)")
plt.xlabel(r"time (ms)")

# make pretty
ax.axis('off')
plt.tight_layout()

# save figure to file
plt.savefig(f'results/3D_new_roi_ECS_K.svg', format='svg')
plt.savefig(f'results/3D_new_roi_ECS_K.png', format='png')

fig = plt.figure(figsize=(15, 10))
ax = plt.gca()

phi_M_norm = get_normalized_phi_M(phi_M)

indices = [i for i, x in enumerate(phi_M_norm) if (x > 0.499 and x < 0.501)]

print(indices)
print("time constant 3D", indices[0]*dt*save_frequency)

t_normalized = np.arange(stimuli_end * dt * save_frequency, Tstop, dt * save_frequency)

ax1 = fig.add_subplot(2,3,1)
plt.plot(t, K_ECS, linewidth=lw, color=pink)
plt.ylabel(r"$\rm c_{K_e}$ (mM)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(2,3,2)
plt.plot(t, E_K, linewidth=lw, color=pink)
plt.ylabel(r"$\rm E_{K}$ (mV)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(2,3,3)
plt.ylabel(r"$\rm I_{Kir}$ ($\rm \mu A/cm^2$)")
plt.plot(t, I_Kir, linewidth=lw, color=pink)
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(2,3,4)
plt.ylabel(r"$\rm c_{K_i}$ (mM)")
plt.plot(t, K_ICS, linewidth=lw, color=pink)
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(2,3,5)
plt.plot(t, phi_M, linewidth=lw, color=pink)
plt.axvline(x=102, color='red', linestyle='--', linewidth=lw*1.2)
plt.ylabel(r"$\rm\phi_M$ (mV)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(2,3,6, xlim=[98, 305])
plt.plot(t_normalized, phi_M_norm, linewidth=lw, color=pink)
plt.plot([stimuli_end * dt * save_frequency, Tstop], [0.5, 0.5], color='grey', linestyle="dotted", linewidth=lw*1.2)
plt.ylabel(r"normalized $\rm\phi_M$")
plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.xlabel(r"time (ms)")

# make pretty
ax.axis('off')
plt.tight_layout()

# save figure to file
plt.savefig(f'results/3D_new_roi.svg', format='svg')
plt.savefig(f'results/3D_new_roi.png', format='png')

"""
fig = plt.figure(figsize=(20, 10))
ax = plt.gca()

alpha_i = 0.11
alpha_e = 0.22
gamma_m = 4.33e4 # 1/cm

ri  = 1 / (sigma_i * alpha_i)  # intracellular resistance k Ohm cm
re  = 1 / (sigma_e * alpha_e) # extracellular resistance k Ohm cm
rm = 1 / (g_tot * gamma_m)  # membrane resistance k Ohm cm**3

length_constant = np.sqrt(rm / (ri + re))  # cm

ax1 = fig.add_subplot(2,4,1)
plt.plot(t, length_constant * 1.0e4, linewidth=lw, color=pink) # convert from cm to um
plt.ylabel(r"Length constant theoretical ($\mu \rm{m}$)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(2,4,2)
plt.plot(t, rm * 1.0e3, linewidth=lw, color=blue) # convert from k Ohm cm**3 to Ohm cm**3
plt.ylabel(r"r_m ($\Omega$ cm)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(2,4,3)
plt.plot(t, ri * 1.0e3, linewidth=lw, color=blue) # convert from k Ohm cm to Ohm cm
plt.ylabel(r"r_i ($\Omega$ cm)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(2,4,4)
plt.plot(t, re * 1.0e3, linewidth=lw, color=blue) # convert from k Ohm cm to Ohm cm
plt.ylabel(r"r_e ($\Omega$ cm)")
plt.xlabel(r"time (ms)")

# make pretty
ax.axis('off')
plt.tight_layout()

# save figure to file
plt.savefig(f'3D_space_constant.svg', format='svg')
plt.savefig(f'3D_space_constant.png', format='png')
"""
