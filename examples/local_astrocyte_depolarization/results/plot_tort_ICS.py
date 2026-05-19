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
        #return x[:240]

fdirs = "ICS-tort-x13"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_I1 = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_I1 = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_I1 = read_me(fname)
fname = f"{fdirs}/i_kir_glial.txt"
I_Kir_I1 = read_me(fname)
fname = f"{fdirs}/E_Cl_glial.txt"
E_Cl_I1 = read_me(fname)
fname = f"{fdirs}/E_Na_glial.txt"
E_Na_I1 = read_me(fname)
fname = f"{fdirs}/E_K_glial.txt"
E_K_I1 = read_me(fname)
fname = f"{fdirs}/g_tot_glial.txt"
g_tot_I1 = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i_I1 = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e_I1 = read_me(fname)

fdirs = "ICS-tort-x31"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_I2 = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_I2 = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_I2 = read_me(fname)
fname = f"{fdirs}/i_kir_glial.txt"
I_Kir_I2 = read_me(fname)
fname = f"{fdirs}/E_Cl_glial.txt"
E_Cl_I2 = read_me(fname)
fname = f"{fdirs}/E_Na_glial.txt"
E_Na_I2 = read_me(fname)
fname = f"{fdirs}/E_K_glial.txt"
E_K_I2 = read_me(fname)
fname = f"{fdirs}/g_tot_glial.txt"
g_tot_I2 = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i_I2 = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e_I2 = read_me(fname)

fdirs = "ICS-tort-x5"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_I3 = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_I3 = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_I3 = read_me(fname)
fname = f"{fdirs}/i_kir_glial.txt"
I_Kir_I3 = read_me(fname)
fname = f"{fdirs}/E_Cl_glial.txt"
E_Cl_I3 = read_me(fname)
fname = f"{fdirs}/E_Na_glial.txt"
E_Na_I3 = read_me(fname)
fname = f"{fdirs}/E_K_glial.txt"
E_K_I3 = read_me(fname)
fname = f"{fdirs}/g_tot_glial.txt"
g_tot_I3 = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i_I3 = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e_I3 = read_me(fname)

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
fname = f"{fdirs}/g_tot_glial.txt"
g_tot_bs = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i_bs = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e_bs = read_me(fname)

# time
dt = 0.1
save_frequency = 5
Tstop = 300
t = np.arange(0, Tstop, dt * save_frequency)

# get index of max value (i.e. where the stimuli is turned off) - same for all
# model variations
stimuli_end = np.argmax(phi_M_I1) + 20
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

phi_M_norm_bs = get_normalized_phi_M(phi_M_bs)
phi_M_norm_I1 = get_normalized_phi_M(phi_M_I1)
phi_M_norm_I2 = get_normalized_phi_M(phi_M_I2)
phi_M_norm_I3 = get_normalized_phi_M(phi_M_I3)

#print(phi_M_norm_I2)
#print(phi_M_norm_I1)
#print(phi_M_norm_bs)

#exit(0)

indices_bs = [i for i, x in enumerate(phi_M_norm_bs) if (x > 0.498 and x < 0.502)]
indices_I1 = [i for i, x in enumerate(phi_M_norm_I1) if (x > 0.498 and x < 0.502)]
indices_I2 = [i for i, x in enumerate(phi_M_norm_I2) if (x > 0.498 and x < 0.502)]
indices_I3 = [i for i, x in enumerate(phi_M_norm_I3) if (x > 0.498 and x < 0.502)]

print(indices_I1)
print(indices_I2)
print(indices_I3)
print(indices_bs)
print("time constant bs", indices_bs[0]*dt*save_frequency)
print("time constant I", indices_I1[0]*dt*save_frequency)
print("time constant E", indices_I2[0]*dt*save_frequency)
print("time constant E", indices_I3[0]*dt*save_frequency)

t_normalized = np.arange(stimuli_end * dt * save_frequency, Tstop, dt * save_frequency)

alpha_i = 0.11
alpha_e = 0.22
gamma_m = 4.33e4 # 1/cm

ri_bs  = 1 / (sigma_i_bs * alpha_i)     # intracellular resistance k Ohm cm
re_bs  = 1 / (sigma_e_bs * alpha_e)     # extracellular resistance k Ohm cm
rm_bs = 1 / (g_tot_bs * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_bs = np.sqrt(rm_bs / (ri_bs + re_bs)) * 1.0e4  # cm

ri_I1  = 1 / (sigma_i_I1 * alpha_i)     # intracellular resistance k Ohm cm
re_I1  = 1 / (sigma_e_I1 * alpha_e)     # extracellular resistance k Ohm cm
rm_I1 = 1 / (g_tot_I1 * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_I1 = np.sqrt(rm_I1 / (ri_I1 + re_I1)) * 1.0e4 # cm

ri_I2  = 1 / (sigma_i_I2 * alpha_i)     # intracellular resistance k Ohm cm
re_I2  = 1 / (sigma_e_I2 * alpha_e)     # extracellular resistance k Ohm cm
rm_I2 = 1 / (g_tot_I2 * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_I2 = np.sqrt(rm_I2 / (ri_I2 + re_I2))  * 1.0e4 # cm

ri_I3  = 1 / (sigma_i_I3 * alpha_i)     # intracellular resistance k Ohm cm
re_I3  = 1 / (sigma_e_I3 * alpha_e)     # extracellular resistance k Ohm cm
rm_I3 = 1 / (g_tot_I3 * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_I3 = np.sqrt(rm_I3 / (ri_I3 + re_I3))  * 1.0e4 # cm

layout = [
    ['A', 'A', 'A'],
    ['B', 'C', 'D']
]

lw = 4

ls_1 = '-'
ls_2 = '--'
ls_3 = '-.'
ls_4 = ':'

fig, axd = plt.subplot_mosaic(layout, figsize=(15, 10), constrained_layout=True)

axd['A'].plot(t, phi_M_I3, linewidth=lw, linestyle=ls_1, color=blue_dark, label=r"$\lambda_i \times 4.4$")
axd['A'].plot(t, phi_M_I2, linewidth=lw, linestyle=ls_2, color=blue, label=r"$\lambda_i \times 3.1$")
axd['A'].plot(t, phi_M_I1, linewidth=lw, linestyle=ls_3, color=blue_light, label=r'$\lambda_i \times 1.3$')
axd['A'].plot(t, phi_M_bs, linewidth=lw*1.4, linestyle=ls_4, color=pink, label=r'baseline')
axd['A'].axvline(x=102, color='red', linestyle='--', linewidth=lw*1.2)
axd['A'].set_ylabel(r"$\phi_M$ (mV)")
axd['A'].set_xlabel(r"time (ms)")
axd['A'].legend()

axd['B'].plot(t, K_ICS_I3, linewidth=lw, linestyle=ls_1, color=blue_dark)
axd['B'].plot(t, K_ICS_I2, linewidth=lw, linestyle=ls_2, color=blue)
axd['B'].plot(t, K_ICS_I1, linewidth=lw, linestyle=ls_3, color=blue_light)
axd['B'].plot(t, K_ICS_bs, linewidth=lw*1.4, linestyle=ls_4, color=pink)
axd['B'].set_ylabel(r"$c_{K_g}$ (mM)")
axd['B'].set_xlabel(r"time (ms)")

axd['C'].plot(t_normalized, phi_M_norm_I3, linewidth=lw, linestyle=ls_1, color=blue_dark)
axd['C'].plot(t_normalized, phi_M_norm_I2, linewidth=lw, linestyle=ls_2, color=blue)
axd['C'].plot(t_normalized, phi_M_norm_I1, linewidth=lw, linestyle=ls_3, color=blue_light)
axd['C'].plot(t_normalized, phi_M_norm_bs, linewidth=lw*1.4, linestyle=ls_4, color=pink)
axd['C'].plot([stimuli_end * dt * save_frequency, Tstop], [0.5, 0.5], color='grey', linestyle="dotted", linewidth=lw*1.2)
axd['C'].set_ylabel(r"normalized $\phi_M$")
axd['C'].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
axd['C'].set_xticks([100, 150, 200, 250, 300])
axd['C'].set_xlabel(r"time (ms)")

axd['D'].plot(t, length_constant_I3, linewidth=lw, linestyle=ls_1, color=blue_dark)
axd['D'].plot(t, length_constant_I2, linewidth=lw, linestyle=ls_2, color=blue)
axd['D'].plot(t, length_constant_I1, linewidth=lw, linestyle=ls_3, color=blue_light)
axd['D'].plot(t, length_constant_bs, linewidth=lw*1.4, linestyle=ls_4, color=pink) 
axd['D'].set_ylabel(r"Length constant theoretical ($\mu$m)")
axd['D'].set_xlabel(r"time (ms)")

print("mean length constant I3:", np.mean(length_constant_I3))
print("mean length constant I2:", np.mean(length_constant_I2))
print("mean length constant I1:", np.mean(length_constant_I1))
print("mean length constant bs:", np.mean(length_constant_bs))

print("max constant I3:", np.max(length_constant_I3))
print("max constant I2:", np.max(length_constant_I2))
print("max constant I1:", np.max(length_constant_I1))
print("max constant bs:", np.max(length_constant_bs))

print("min constant I3:", np.min(length_constant_I3))
print("min constant I2:", np.min(length_constant_I2))
print("min constant I1:", np.min(length_constant_I1))
print("min constant bs:", np.min(length_constant_bs))

print("-------------------------------")

print("max K ICS I3:", np.max(K_ICS_I3) - np.max(K_ICS_bs))
print("max K ICS I2:", np.max(K_ICS_I2) - np.max(K_ICS_bs))
print("max K ICS I1:", np.max(K_ICS_I1) - np.max(K_ICS_bs))

print("-------------------------------")

print("max K phi bs:", np.max(phi_M_bs))
print("max K phi I1:", np.max(phi_M_I1))
print("max K phi I2:", np.max(phi_M_I2))
print("max K phi I3:", np.max(phi_M_I3))

print("-------------------------------")

print(max(length_constant_bs))
print(max(length_constant_I1))
print(max(length_constant_I2))
print(max(length_constant_I3))

# make pretty
plt.tight_layout()

# save figure to file
plt.savefig(f'tort_ICS.svg', format='svg')
plt.savefig(f'tort_ICS.png', format='png')
