import numpy as np
import sys

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

#blue = '#3d83c4'
blue = "#0000FF"
green = "#057a69"
green_light = '#63cf32'

grey = "#423c3c"
pink = '#e31be3'
blue_light = "#56B4E9"
blue_dark = "#191970"
blue = "#3975db"

pink_tt = '#e31be330'
green_tt = "#057a6930"

pink_t = '#e31be380'
green_t = "#057a6980"



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

fdirs = "1D"
fname = f"{fdirs}/phi_M.txt"
phi_M_1D = read_me(fname)
fname = f"{fdirs}/K_ECS.txt"
K_ECS_1D = read_me(fname)
fname = f"{fdirs}/K_ICS.txt"
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
fname = f"{fdirs}/g_tot.txt"
g_tot_1D = read_me(fname)
fname = f"{fdirs}/sigma_i.txt"
sigma_i_1D = read_me(fname)
fname = f"{fdirs}/sigma_e.txt"
sigma_e_1D = read_me(fname)

fname = f"{fdirs}/phi_M_global.txt"
phi_M_global_1D = read_me(fname)
fname = f"{fdirs}/phi_M_roi.txt"
phi_M_roi_1D = read_me(fname)
fname = f"{fdirs}/K_E_global.txt"
K_E_global_1D = read_me(fname)
fname = f"{fdirs}/K_E_roi.txt"
K_E_roi_1D = read_me(fname)
fname = f"{fdirs}/K_G_global.txt"
K_G_global_1D = read_me(fname)
fname = f"{fdirs}/K_G_roi.txt"
K_G_roi_1D = read_me(fname)
fname = f"{fdirs}/E_K_global.txt"
E_K_global_1D = read_me(fname)
fname = f"{fdirs}/E_K_roi.txt"
E_K_roi_1D = read_me(fname)
fname = f"{fdirs}/I_Kir_global.txt"
I_Kir_global_1D = read_me(fname)
fname = f"{fdirs}/I_Kir_roi.txt"
I_Kir_roi_1D = read_me(fname)

# get phi_M time
fdirs = "baseline"
fname = f"{fdirs}/phi_M_glial.txt"
phi_M_3D = read_me(fname)
fname = f"{fdirs}/K_ECS_glial.txt"
K_ECS_3D = read_me(fname)
fname = f"{fdirs}/K_ICS_glial.txt"
K_ICS_3D = read_me(fname)
fname = f"{fdirs}/E_Cl_glial.txt"
E_Cl_3D = read_me(fname)
fname = f"{fdirs}/E_Na_glial.txt"
E_Na_3D = read_me(fname)
fname = f"{fdirs}/E_K_glial.txt"
E_K_3D = read_me(fname)
fname = f"{fdirs}/i_kir_glial.txt"
I_Kir_3D = read_me(fname)
fname = f"{fdirs}/i_pump_glial.txt"
I_pump_3D = read_me(fname)
fname = f"{fdirs}/g_tot_glial.txt"
g_tot_3D = read_me(fname)
fname = f"{fdirs}/sigma_i_glial.txt"
sigma_i_3D = read_me(fname)
fname = f"{fdirs}/sigma_e_glial.txt"
sigma_e_3D = read_me(fname)

# time
dt = 0.1
save_frequency = 5
Tstop = 300
t = np.arange(0, Tstop, dt * save_frequency)

# get index of max value (i.e. where the stimuli is turned off) - same for all
# model variations
stimuli_end = np.argmax(phi_M_1D) + 20
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

fig = plt.figure(figsize=(15, 15))
ax = plt.gca()

phi_M_norm_1D = get_normalized_phi_M(phi_M_1D)
phi_M_norm_3D = get_normalized_phi_M(phi_M_3D)

indices_3D = [i for i, x in enumerate(phi_M_norm_3D) if (x > 0.499 and x < 0.501)]
indices_1D = [i for i, x in enumerate(phi_M_norm_1D) if (x > 0.499 and x < 0.501)]

print(indices_1D)
print(indices_3D)
print("time constant 1D", indices_1D[0]*dt*save_frequency)
print("time constant 3D", indices_3D[0]*dt*save_frequency)

t_normalized = np.arange(stimuli_end * dt * save_frequency, Tstop, dt * save_frequency)

phi_M_space_norm = get_normalized_phi_M_space(phi_M_space)
t_normalized_space = np.linspace(100, 200, int(3200/2))
x = np.linspace(0, 200, 3200)

ax1 = fig.add_subplot(3,3,1)
plt.plot(t, K_ECS_3D, linewidth=lw, color=pink, label=r"3D")
plt.plot(t, K_ECS_1D, linewidth=lw, color=green, label=r'1D')
plt.plot(t, K_E_roi_1D, linewidth=lw, color=green_light, linestyle="dotted", label=r'avg.~1D roi')
plt.plot(t, K_E_global_1D, linewidth=lw, color=green_light, linestyle="dashed", label=r'avg.~1D')
plt.ylabel(r"$\rm c_{K_e}$ (mM)")
plt.xlabel(r"time (ms)")
plt.legend()

ax1 = fig.add_subplot(3,3,2)
plt.plot(t, E_K_3D, linewidth=lw, color=pink)
plt.plot(t, E_K_1D, linewidth=lw, color=green)
plt.plot(t, E_K_roi_1D, linewidth=lw, color=green_light,    linestyle="dotted")
plt.plot(t, E_K_global_1D, linewidth=lw, color=green_light, linestyle="dashed")
plt.ylabel(r"$\rm E_{K}$ (mV)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(3,3,3)
plt.ylabel(r"$\rm I_{Kir}$ ($\rm \mu A/cm^2$)")
plt.plot(t, I_Kir_3D, linewidth=lw, color=pink)
plt.plot(t, I_Kir_1D, linewidth=lw, color=green)
plt.plot(t, I_Kir_roi_1D, linewidth=lw, color=green_light,    linestyle="dotted")
plt.plot(t, I_Kir_global_1D, linewidth=lw, color=green_light, linestyle="dashed")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(3,3,4)
plt.plot(t, K_ICS_3D, linewidth=lw, color=pink)
plt.plot(t, K_ICS_1D, linewidth=lw, color=green)
plt.plot(t, K_G_roi_1D, linewidth=lw, color=green_light,    linestyle="dotted")
plt.plot(t, K_G_global_1D, linewidth=lw, color=green_light, linestyle="dashed")
plt.ylabel(r"$\rm c_{K_g}$ (mM)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(3,3,5)
plt.plot(t, phi_M_3D, linewidth=lw, color=pink)
plt.plot(t, phi_M_1D*1.0e3, linewidth=lw, color=green)
plt.plot(t, phi_M_global_1D*1.0e3, linewidth=lw, color=green_light,linestyle="dashed")
plt.plot(t, phi_M_roi_1D*1.0e3, linewidth=lw, color=green_light, linestyle="dotted")
plt.axvline(x=102, color='red', linestyle='--', linewidth=lw*1.2)
plt.ylabel(r"$\rm\phi_M$ (mV)")
plt.xlabel(r"time (ms)")

ax1 = fig.add_subplot(3,3,6, xlim=[98, 305])
plt.plot(t_normalized, phi_M_norm_3D, linewidth=lw, color=pink, label=r"3D")
plt.plot(t_normalized, phi_M_norm_1D, linewidth=lw, color=green, label=r'1D')
plt.plot([stimuli_end * dt * save_frequency, Tstop], [0.5, 0.5], color='grey', linestyle="dotted", linewidth=lw*1.2)
plt.ylabel(r"normalized $\rm \phi_M$")
plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.xlabel(r"time (ms)")

# Calculate stuff for resistance and length constant
alpha_i = 0.11
alpha_e = 0.22
gamma_m_3D = 4.33e4 # 1/cm
gamma_m_1D = 4.33e6 # 1/m

ri_1D  = 1 / (sigma_i_1D * alpha_i)  # intracellular resistance Ohm m
re_1D  = 1 / (sigma_e_1D * alpha_e)  # extracellular resistance Ohm m
rm_1D = 1 / (g_tot_1D * gamma_m_1D)  # membrane resistance Ohm m**3

ri_3D  = 1 / (sigma_i_3D * alpha_i)  # intracellular resistance k Ohm cm
re_3D  = 1 / (sigma_e_3D * alpha_e)  # extracellular resistance k Ohm cm
rm_3D = 1 / (g_tot_3D * gamma_m_3D)  # membrane resistance k Ohm cm**3

# Calculate length constant (taking into account both variations in ECS and ICS space)
length_constant_1D = np.sqrt(rm_1D / (ri_1D + re_1D))  # m
length_constant_3D = np.sqrt(rm_3D / (ri_3D + re_3D))  # cm

# Calculate length constant in the "classical way", that is do not take into
# account the ECS space
length_constant_no_ECS = np.sqrt(rm_1D / ri_1D) # m
# Calculate length constant using formula from Halnes et al 2013 (eq 16)
length_constant_no_ECS_Halnes_formula = np.sqrt(alpha_i * sigma_i_1D / (gamma_m_1D * g_tot_1D))  # m

ax1 = fig.add_subplot(3,3,7)
plt.plot(x, phi_M_space*1.0e3, linewidth=lw, color=green, label=r'1D')
plt.ylabel(r"$\rm \phi_M$ (mV)")
plt.xticks([0, 50, 100, 150, 200])
plt.xlabel(r"$x(\mu\rm{m})$")

ax1 = fig.add_subplot(3,3,8)
plt.plot(t_normalized_space, phi_M_space_norm, linewidth=lw, color=green, label=r'1D')
plt.plot([100, 200], [0.36787944117, 0.36787944117], color='grey', linestyle="dotted", linewidth=lw)
plt.ylabel(r"normalized $\phi_M$")
plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.xticks([100, 125, 150, 175, 200])
plt.xlabel(r"$x(\mu\rm{m})$")

#print(phi_M_space_norm)
x = 0.36787944117
indices = np.where(np.isclose(phi_M_space_norm, x, atol=0.001))[0][0]
x_value = indices*6.25e-2
print(f"length_constant 1D is: {x_value} um")

print("--------------------------------------------------------------")
print(f"length_constant mean 1D is: {np.mean(length_constant_1D)*1.0e6} um")
print(f"length_constant max 1D is: {np.max(length_constant_1D)*1.0e6} um")
print(f"length_constant min 1D is: {np.min(length_constant_1D)*1.0e6} um")

print("--------------------------------------------------------------")
print(f"length_constant mean 3D is: {np.mean(length_constant_3D)*1.0e4} um")
print(f"length_constant max 3D is: {np.max(length_constant_3D)*1.0e4} um")
print(f"length_constant min 3D is: {np.min(length_constant_3D)*1.0e4} um")
print("--------------------------------------------------------------")

ax1 = fig.add_subplot(3,3,9)
plt.plot(t, length_constant_1D * 1.0e6, linewidth=lw, color=green, label=r"1D")
plt.plot(t, length_constant_3D * 1.0e4, linewidth=lw, color=pink, label=r"3D")
plt.ylabel(r"theoretical length constant ($\mu$m)")
plt.xlabel(r"time (ms)")

# make pretty
ax.axis('off')
plt.tight_layout()

# save figure to file
plt.savefig(f'1D.svg', format='svg')
plt.savefig(f'1D.png', format='png')

# Make bar-plot with mean values

K_stim_1D = np.mean(K_ECS_1D[:stimuli_end])
K_stim_3D = np.mean(K_ECS_3D[:stimuli_end])
K_decay_1D = np.mean(K_ECS_1D[stimuli_end:])
K_decay_3D = np.mean(K_ECS_3D[stimuli_end:])
K_tot_1D = np.mean(K_ECS_1D)
K_tot_3D = np.mean(K_ECS_3D)

E_K_stim_1D = np.mean(E_K_1D[:stimuli_end])
E_K_stim_3D = np.mean(E_K_3D[:stimuli_end])
E_K_decay_1D = np.mean(E_K_1D[stimuli_end:])
E_K_decay_3D = np.mean(E_K_3D[stimuli_end:])
E_K_tot_1D = np.mean(E_K_1D)
E_K_tot_3D = np.mean(E_K_3D)

I_Kir_stim_1D = np.mean(I_Kir_1D[:stimuli_end])
I_Kir_stim_3D = np.mean(I_Kir_3D[:stimuli_end])
I_Kir_decay_1D = np.mean(I_Kir_1D[stimuli_end:])
I_Kir_decay_3D = np.mean(I_Kir_3D[stimuli_end:])
I_Kir_tot_1D = np.mean(I_Kir_1D)
I_Kir_tot_3D = np.mean(I_Kir_3D)

categories = [r'stimuli', r'decay', r'total']

h1_1D = [K_stim_1D, K_decay_1D, K_tot_1D]
h1_3D = [K_stim_3D, K_decay_3D, K_tot_3D]

h2_1D = [E_K_stim_1D, E_K_decay_1D, E_K_tot_1D]
h2_3D = [E_K_stim_3D, E_K_decay_3D, E_K_tot_3D]

h3_1D = [I_Kir_stim_1D, I_Kir_decay_1D, I_Kir_tot_1D]
h3_3D = [I_Kir_stim_3D, I_Kir_decay_3D, I_Kir_tot_3D]

colors_1D_h = [green, green, green]
colors_3D_h = [pink, pink, pink]

# 2. Set the positions of the bars on the x-axis
x = np.arange(len(categories)) 
width = 0.4  # Width of each bar

fig, ax = plt.subplots(figsize=(8, 5))
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# 3. Define each subplot explicitly
# Top Left
axes[0].bar(x - width/2, h1_1D, width, color=colors_1D_h, edgecolor='black')
axes[0].bar(x + width/2, h1_3D, width, color=colors_3D_h, edgecolor='black')
axes[0].set_ylabel(r'Mean $[\rm{K}^+]_e$ (mM)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories)

# Top Right
axes[1].bar(x - width/2, h2_1D, width, label=r'1D', color=colors_1D_h, edgecolor='black')
axes[1].bar(x + width/2, h2_3D, width, label=r'3D', color=colors_3D_h, edgecolor='black')
axes[1].set_ylabel(r'Mean $\rm E_K$ (mV)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories)

# Bottom Left
axes[2].bar(x - width/2, h3_1D, width, label=r'1D', color=colors_1D_h, edgecolor='black')
axes[2].bar(x + width/2, h3_3D, width, label=r'3D', color=colors_3D_h, edgecolor='black')
axes[2].set_ylabel(r"Mean $\rm I_{Kir}$ ($\rm \mu A/cm^2$)")
axes[2].set_xticks(x)
axes[2].set_xticklabels(categories)
axes[2].set_xticklabels(categories)
axes[2].legend()

# 4. Final layout touch
plt.tight_layout()
# 4. Clean up layout to prevent labels from overlapping
plt.savefig(f'1D_histrogram.png', format='png')
plt.savefig(f'1D_histrogram.svg', format='svg')
plt.close()
