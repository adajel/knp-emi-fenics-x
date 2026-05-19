import numpy as np
import sys

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

grey = "#423c3c"
green = "#057a69"
pink = '#e31be3'

blue_light = "#56B4E9"
blue_dark = "#191970"
blue = "#3975db"

blue_light_t = "#56B4E980"
blue_dark_t = "#19197080"
blue_t = "#3975db80"

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

fdirs = "ECS-tort-x13"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_E1 = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_E1 = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_E1 = read_me(fname)
fname = f"{fdirs}/i_kir_glial.txt"
I_Kir_E1 = read_me(fname)
fname = f"{fdirs}/E_Cl_glial.txt"
E_Cl_E1 = read_me(fname)
fname = f"{fdirs}/E_Na_glial.txt"
E_Na_E1 = read_me(fname)
fname = f"{fdirs}/E_K_glial.txt"
E_K_E1 = read_me(fname)
fname = f"{fdirs}/g_tot_glial.txt"
g_tot_E1 = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i_E1 = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e_E1 = read_me(fname)

fdirs = "ECS-tort-x31"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_E2 = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_E2 = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_E2 = read_me(fname)
fname = f"{fdirs}/i_kir_glial.txt"
I_Kir_E2 = read_me(fname)
fname = f"{fdirs}/E_Cl_glial.txt"
E_Cl_E2 = read_me(fname)
fname = f"{fdirs}/E_Na_glial.txt"
E_Na_E2 = read_me(fname)
fname = f"{fdirs}/E_K_glial.txt"
E_K_E2 = read_me(fname)
fname = f"{fdirs}/g_tot_glial.txt"
g_tot_E2 = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i_E2 = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e_E2 = read_me(fname)

fdirs = "ECS-tort-x5"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_E3 = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_E3 = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_E3 = read_me(fname)
fname = f"{fdirs}/i_kir_glial.txt"
I_Kir_E3 = read_me(fname)
fname = f"{fdirs}/E_Cl_glial.txt"
E_Cl_E3 = read_me(fname)
fname = f"{fdirs}/E_Na_glial.txt"
E_Na_E3 = read_me(fname)
fname = f"{fdirs}/E_K_glial.txt"
E_K_E3 = read_me(fname)
fname = f"{fdirs}/g_tot_glial.txt"
g_tot_E3 = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i_E3 = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e_E3 = read_me(fname)

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
stimuli_end = np.argmax(phi_M_E1) + 20
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
phi_M_norm_E1 = get_normalized_phi_M(phi_M_E1)
phi_M_norm_E2 = get_normalized_phi_M(phi_M_E2)
phi_M_norm_E3 = get_normalized_phi_M(phi_M_E3)

#print(phi_M_norm_E2)
#print(phi_M_norm_E1)
#print(phi_M_norm_bs)

#exit(0)

indices_bs = [i for i, x in enumerate(phi_M_norm_bs) if (x > 0.498 and x < 0.502)]
indices_E1 = [i for i, x in enumerate(phi_M_norm_E1) if (x > 0.498 and x < 0.502)]
indices_E2 = [i for i, x in enumerate(phi_M_norm_E2) if (x > 0.498 and x < 0.502)]
indices_E3 = [i for i, x in enumerate(phi_M_norm_E3) if (x > 0.498 and x < 0.502)]

print(indices_E1)
print(indices_E2)
print(indices_E3)
print(indices_bs)
print("time constant bs", indices_bs[0]*dt*save_frequency)
print("time constant I", indices_E1[0]*dt*save_frequency)
print("time constant E", indices_E2[0]*dt*save_frequency)
print("time constant E", indices_E3[0]*dt*save_frequency)

t_normalized = np.arange(stimuli_end * dt * save_frequency, Tstop, dt * save_frequency)

alpha_i = 0.11
alpha_e = 0.22
gamma_m = 4.33e4 # 1/cm

ri_bs  = 1 / (sigma_i_bs * alpha_i)     # intracellular resistance k Ohm cm
re_bs  = 1 / (sigma_e_bs * alpha_e)     # extracellular resistance k Ohm cm
rm_bs = 1 / (g_tot_bs * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_bs = np.sqrt(rm_bs / (ri_bs + re_bs)) * 1.0e4  # cm

rfdirs = "ECS-tort-x13"
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

ri_E1  = 1 / (sigma_i_E1 * alpha_i)     # intracellular resistance k Ohm cm
re_E1  = 1 / (sigma_e_E1 * alpha_e)     # extracellular resistance k Ohm cm
rm_E1 = 1 / (g_tot_E1 * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_E1 = np.sqrt(rm_E1 / (ri_E1 + re_E1)) * 1.0e4 # cm

ri_E2  = 1 / (sigma_i_E2 * alpha_i)     # intracellular resistance k Ohm cm
re_E2  = 1 / (sigma_e_E2 * alpha_e)     # extracellular resistance k Ohm cm
rm_E2 = 1 / (g_tot_E2 * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_E2 = np.sqrt(rm_E2 / (ri_E2 + re_E2))  * 1.0e4 # cm

ri_E3  = 1 / (sigma_i_E3 * alpha_i)     # intracellular resistance k Ohm cm
re_E3  = 1 / (sigma_e_E3 * alpha_e)     # extracellular resistance k Ohm cm
rm_E3 = 1 / (g_tot_E3 * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_E3 = np.sqrt(rm_E3 / (ri_E3 + re_E3))  * 1.0e4 # cm

layout = [
    ['A', 'A', 'A'],
    ['B', 'B', 'B'],
    ['C', 'D', 'E']
]

lw = 4

ls_1 = '-'
ls_2 = '--'
ls_3 = '-.'
ls_4 = ':'

fig, axd = plt.subplot_mosaic(layout, figsize=(15, 15), constrained_layout=True)

axd['A'].plot(t, K_ECS_E3, linewidth=lw, linestyle=ls_1, color=blue_dark, label=r"$\lambda_e \times 4.4$")
axd['A'].plot(t, K_ECS_E2, linewidth=lw, linestyle=ls_2, color=blue, label=r"$\lambda_e \times 3.1$")
axd['A'].plot(t, K_ECS_E1, linewidth=lw, linestyle=ls_3, color=blue_light, label=r'$\lambda_e \times 1.3$')
axd['A'].plot(t, K_ECS_bs, linewidth=lw*1.4, linestyle=ls_4, color=pink, label=r'baseline')
axd['A'].set_ylabel(r"$[\rm K]_e$ (mM)")
axd['A'].set_xlabel(r"time (ms)")
axd['A'].legend()

axd['B'].plot(t, phi_M_E3, linewidth=lw, linestyle=ls_1, color=blue_dark)
axd['B'].plot(t, phi_M_E2, linewidth=lw, linestyle=ls_2, color=blue)
axd['B'].plot(t, phi_M_E1, linewidth=lw, linestyle=ls_3, color=blue_light)
axd['B'].plot(t, phi_M_bs, linewidth=lw, linestyle=ls_4, color=pink)
axd['B'].axvline(x=102, color='red', linestyle='--', linewidth=lw*1.2)
axd['B'].set_ylabel(r"$\phi_M$ (mV)")
axd['B'].set_xlabel(r"time (ms)")

axd['C'].plot(t, K_ICS_E3, linewidth=lw, linestyle=ls_1, color=blue_dark)
axd['C'].plot(t, K_ICS_E2, linewidth=lw, linestyle=ls_2, color=blue)
axd['C'].plot(t, K_ICS_E1, linewidth=lw, linestyle=ls_3, color=blue_light)
axd['C'].plot(t, K_ICS_bs, linewidth=lw, linestyle=ls_4, color=pink)
axd['C'].set_ylabel(r"$c_{K_g}$ (mM)")
axd['C'].set_xlabel(r"time (ms)")

axd['D'].plot(t_normalized, phi_M_norm_E3, linewidth=lw, linestyle=ls_1, color=blue_dark)
axd['D'].plot(t_normalized, phi_M_norm_E2, linewidth=lw, linestyle=ls_2, color=blue)
axd['D'].plot(t_normalized, phi_M_norm_E1, linewidth=lw, linestyle=ls_3, color=blue_light)
axd['D'].plot(t_normalized, phi_M_norm_bs, linewidth=lw, linestyle=ls_4, color=pink)
axd['D'].plot([stimuli_end * dt * save_frequency, Tstop], [0.5, 0.5], color='grey', linestyle="dotted", linewidth=lw*1.2)
axd['D'].set_ylabel(r"normalized $\phi_M$")
axd['D'].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
axd['D'].set_xticks([100, 150, 200, 250, 300])
axd['D'].set_xlabel(r"time (ms)")

axd['E'].plot(t, length_constant_E3, linewidth=lw, linestyle=ls_1, color=blue_dark)
axd['E'].plot(t, length_constant_E2, linewidth=lw, linestyle=ls_2, color=blue)
axd['E'].plot(t, length_constant_E1, linewidth=lw, linestyle=ls_3, color=blue_light)
axd['E'].plot(t, length_constant_bs, linewidth=lw, linestyle=ls_4, color=pink)
axd['E'].set_ylabel(r"Length constant theoretical ($\mu$m)")
axd['E'].set_xlabel(r"time (ms)")

print("mean length constant E3:", np.mean(length_constant_E3))
print("mean length constant E2:", np.mean(length_constant_E2))
print("mean length constant E1:", np.mean(length_constant_E1))
print("mean length constant bs:", np.mean(length_constant_bs))

print("max constant E3:", np.max(length_constant_E3))
print("max constant E2:", np.max(length_constant_E2))
print("max constant E1:", np.max(length_constant_E1))
print("max constant bs:", np.max(length_constant_bs))

print("min constant E3:", np.min(length_constant_E3))
print("min constant E2:", np.min(length_constant_E2))
print("min constant E1:", np.min(length_constant_E1))
print("min constant bs:", np.min(length_constant_bs))

print("-------------------------------")

print("max K ICS E3:", np.max(K_ICS_E3) - np.max(K_ICS_bs))
print("max K ICS E2:", np.max(K_ICS_E2) - np.max(K_ICS_bs))
print("max K ICS E1:", np.max(K_ICS_E1) - np.max(K_ICS_bs))

print("-------------------------------")

print("max K phi bs:", np.max(phi_M_bs))
print("max K phi E1:", np.max(phi_M_E1))
print("max K phi E2:", np.max(phi_M_E2))
print("max K phi E3:", np.max(phi_M_E3))

print("-------------------------------")

print(max(length_constant_bs))
print(max(length_constant_E1))
print(max(length_constant_E2))
print(max(length_constant_E3))

# make pretty
plt.tight_layout()

# save figure to file
plt.savefig(f'tort_ECS.svg', format='svg')
plt.savefig(f'tort_ECS.png', format='png')


fdirs = "ICS-tort-x13"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_I1 = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_I1 = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_I1 = read_me(fname)
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
fname = f"{fdirs}/g_tot_glial.txt"
g_tot_I3 = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i_I3 = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e_I3 = read_me(fname)

fdirs = "ECS-tort-x13"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_EI1 = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_EI1 = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_EI1 = read_me(fname)
fname = f"{fdirs}/g_tot_glial.txt"
g_tot_EI1 = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i_EI1 = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e_EI1 = read_me(fname)

fdirs = "ECS-tort-x31"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_EI2 = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_EI2 = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_EI2 = read_me(fname)
fname = f"{fdirs}/g_tot_glial.txt"
g_tot_EI2 = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i_EI2 = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e_EI2 = read_me(fname)

fdirs = "ECS-tort-x5"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_EI3 = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_EI3 = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_EI3 = read_me(fname)
fname = f"{fdirs}/g_tot_glial.txt"
g_tot_EI3 = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i_EI3 = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e_EI3 = read_me(fname)

ri_I1  = 1 / (sigma_i_I1 * alpha_i)     # intracellular resistance k Ohm cm
re_I1  = 1 / (sigma_e_I1 * alpha_e)     # extracellular resistance k Ohm cm
rm_I1 = 1  / (g_tot_I1 * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_I1 = np.sqrt(rm_I1 / (ri_I1 + re_I1)) * 1.0e4 # cm

ri_I2  = 1 / (sigma_i_I2 * alpha_i)     # intracellular resistance k Ohm cm
re_I2  = 1 / (sigma_e_I2 * alpha_e)     # extracellular resistance k Ohm cm
rm_I2 = 1  / (g_tot_I2 * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_I2 = np.sqrt(rm_I2 / (ri_I2 + re_I2))  * 1.0e4 # cm

ri_I3  = 1 / (sigma_i_I3 * alpha_i)     # intracellular resistance k Ohm cm
re_I3  = 1 / (sigma_e_I3 * alpha_e)     # extracellular resistance k Ohm cm
rm_I3 = 1  / (g_tot_I3 * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_I3 = np.sqrt(rm_I3 / (ri_I3 + re_I3))  * 1.0e4 # cm

ri_EI1  = 1 / (sigma_i_EI1 * alpha_i)     # intracellular resistance k Ohm cm
re_EI1  = 1 / (sigma_e_EI1 * alpha_e)     # extracellular resistance k Ohm cm
rm_EI1 = 1  / (g_tot_EI1 * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_EI1 = np.sqrt(rm_EI1 / (ri_EI1 + re_EI1)) * 1.0e4 # cm

ri_EI2  = 1 / (sigma_i_EI2 * alpha_i)     # intracellular resistance k Ohm cm
re_EI2  = 1 / (sigma_e_EI2 * alpha_e)     # extracellular resistance k Ohm cm
rm_EI2 = 1  / (g_tot_EI2 * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_EI2 = np.sqrt(rm_EI2 / (ri_EI2 + re_EI2))  * 1.0e4 # cm

ri_EI3  = 1 / (sigma_i_EI3 * alpha_i)     # intracellular resistance k Ohm cm
re_EI3  = 1 / (sigma_e_EI3 * alpha_e)     # extracellular resistance k Ohm cm
rm_EI3 = 1 / (g_tot_EI3 * gamma_m)        # membrane resistance k Ohm cm**3
length_constant_EI3 = np.sqrt(rm_EI3 / (ri_EI3 + re_EI3))  * 1.0e4 # cm

# 1. Data and Colors
K_E_max_bs = np.max(K_ECS_bs)
# lambda e
K_E_max_E1 = np.max(K_ECS_E1)
K_E_max_E2 = np.max(K_ECS_E2)
K_E_max_E3 = np.max(K_ECS_E3)
# lambda i
K_E_max_I1 = np.max(K_ECS_I1)
K_E_max_I2 = np.max(K_ECS_I2)
K_E_max_I3 = np.max(K_ECS_I3)
# lambda e and i
K_E_max_EI1 = np.max(K_ECS_I1)
K_E_max_EI2 = np.max(K_ECS_I2)
K_E_max_EI3 = np.max(K_ECS_I3)

phi_M_max_bs = np.max(phi_M_bs)
# lambda e
phi_M_max_E1 = np.max(phi_M_E1)
phi_M_max_E2 = np.max(phi_M_E2)
phi_M_max_E3 = np.max(phi_M_E3)
# lambda i
phi_M_max_I1 = np.max(phi_M_I1)
phi_M_max_I2 = np.max(phi_M_I2)
phi_M_max_I3 = np.max(phi_M_I3)
# lambda e and i
phi_M_max_EI1 = np.max(phi_M_EI1)
phi_M_max_EI2 = np.max(phi_M_EI2)
phi_M_max_EI3 = np.max(phi_M_EI3)

lc_bs_mean = np.mean(length_constant_bs)
# lambda i
lc_I1_mean = np.mean(length_constant_I1)
lc_I2_mean = np.mean(length_constant_I2)
lc_I3_mean = np.mean(length_constant_I3)
# lambda e
lc_E1_mean = np.mean(length_constant_E1)
lc_E2_mean = np.mean(length_constant_E2)
lc_E3_mean = np.mean(length_constant_E3)
# lambda e and i
lc_EI1_mean = np.mean(length_constant_EI1)
lc_EI2_mean = np.mean(length_constant_EI2)
lc_EI3_mean = np.mean(length_constant_EI3)

phi_M_norm_bs = get_normalized_phi_M(phi_M_bs)
# lambda i
phi_M_norm_I1 = get_normalized_phi_M(phi_M_I1)
phi_M_norm_I2 = get_normalized_phi_M(phi_M_I2)
phi_M_norm_I3 = get_normalized_phi_M(phi_M_I3)
# lambda e and i
phi_M_norm_EI1 = get_normalized_phi_M(phi_M_EI1)
phi_M_norm_EI2 = get_normalized_phi_M(phi_M_EI2)
phi_M_norm_EI3 = get_normalized_phi_M(phi_M_EI3)

# lambda i
indices_bs = [i for i, x in enumerate(phi_M_norm_bs) if (x > 0.498 and x < 0.502)]
indices_I1 = [i for i, x in enumerate(phi_M_norm_I1) if (x > 0.498 and x < 0.502)]
indices_I2 = [i for i, x in enumerate(phi_M_norm_I2) if (x > 0.498 and x < 0.502)]
indices_I3 = [i for i, x in enumerate(phi_M_norm_I3) if (x > 0.498 and x < 0.502)]
# lambda e and i
indices_EI1 = [i for i, x in enumerate(phi_M_norm_EI1) if (x > 0.498 and x < 0.502)]
indices_EI2 = [i for i, x in enumerate(phi_M_norm_EI2) if (x > 0.498 and x < 0.502)]
indices_EI3 = [i for i, x in enumerate(phi_M_norm_EI3) if (x > 0.498 and x < 0.502)]

tc_bs = indices_bs[0]*dt*save_frequency
# lambda e
tc_E1 = indices_E1[0]*dt*save_frequency
tc_E2 = indices_E2[0]*dt*save_frequency
tc_E3 = indices_E3[0]*dt*save_frequency
# lambda i
tc_I1 = indices_I1[0]*dt*save_frequency
tc_I2 = indices_I2[0]*dt*save_frequency
tc_I3 = indices_I3[0]*dt*save_frequency
# lambda e and i
tc_EI1 = indices_EI1[0]*dt*save_frequency
tc_EI2 = indices_EI2[0]*dt*save_frequency
tc_EI3 = indices_EI3[0]*dt*save_frequency

categories = [r'baseline',
              r'$\times 1.3$',
              r'$\times 3.1$',
              r'$\times 4.4$']

h1, h2, h3, h4 = [K_E_max_bs,
                  K_E_max_E1,
                  K_E_max_E2,
                  K_E_max_E3], \
                 [phi_M_max_bs,
                 phi_M_max_E1,
                 phi_M_max_E2,
                 phi_M_max_E3], \
                 [tc_bs,
                  tc_E1,
                  tc_E2,
                  tc_E3], \
                 [lc_bs_mean,
                  lc_E1_mean,
                  lc_E2_mean,
                  lc_E3_mean]

f1, f2, f3, f4 = [0, \
                  K_E_max_EI1 - K_E_max_E1+2, \
                  K_E_max_EI2 - K_E_max_E2+2, \
                  K_E_max_EI3 - K_E_max_E3+2], \
                 [0, \
                  phi_M_max_EI1 - phi_M_max_E1+2,\
                  phi_M_max_EI2 - phi_M_max_E2+2, \
                  phi_M_max_EI3 - phi_M_max_E3+2], \
                 [0,\
                  tc_EI1 - tc_E1+2, \
                  tc_EI2 - tc_E2+2, \
                  tc_EI3 - tc_E3+2], \
                 [0, \
                  lc_EI1_mean - lc_E1_mean+2, \
                  lc_EI2_mean - lc_E2_mean+2, \
                  lc_EI3_mean - lc_E3_mean+2]

colors_h = [pink, blue_light, blue, blue_dark]
colors_f = [pink, blue_light_t, blue_t, blue_dark_t]

# 2. Create Figure
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 6))

# 3. Define each subplot explicitly
# Top Left
axes[0].bar(categories, h1, color=colors_h, edgecolor='black', width=0.8)
axes[0].bar(categories, f1, color=colors_f, edgecolor='black', width=0.8, bottom=h1)
axes[0].set_ylabel(r'$[\rm{K}^+]_e$ max (mM)')
#axes[0].legend()

# Top Right
axes[1].bar(categories, h2, color=colors_h, edgecolor='black', width=0.8)
axes[1].bar(categories, f2, color=colors_f, edgecolor='black', width=0.8, bottom=h2)
axes[1].set_ylabel(r'$\phi_M$ max (mV)')

# Bottom Left
axes[2].bar(categories, h3, color=colors_h, edgecolor='black', width=0.8)
axes[2].bar(categories, f3, color=colors_f, edgecolor='black', width=0.8, bottom=h3)
axes[2].set_ylabel(r"Time constant (ms)")

# Bottom Right
axes[3].bar(categories, h4, color=colors_h, edgecolor='black', width=0.8)
axes[3].bar(categories, f4, color=colors_f, edgecolor='black', width=0.8, bottom=h4)
axes[3].set_ylabel(r"Mean length constant theoretical ($\mu$m)")

# 4. Final layout touch
plt.tight_layout()
# 4. Clean up layout to prevent labels from overlapping
plt.savefig(f'tort_ECS_histrogram.png', format='png')
plt.close()
