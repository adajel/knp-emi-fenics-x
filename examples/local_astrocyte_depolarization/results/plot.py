import numpy as np
import sys

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

red = '#db2525'
red_light = '#f06c62'
red_dark = '#611b15'

blue = '#1f3ecc'
blue_dark = '#00206b'
blue_light = '#648ded'

# set font & text parameters
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'

#\definecolor{blue1}{RGB}{54, 92, 141}

def read_me(fname):
    with open(fname) as f:
        lines = f.readlines()
        x = [float(line.split()[0]) for line in lines]
    return x

fdir_100 = "local_PAP_depolarization_100_hz"
fdir_300 = "local_PAP_depolarization_100_hz"

#fdir_100 = "baseline"
#fdir_300 = "baseline"

# write phi_M
fname = f"{fdir_100}/phi_M_glial.txt"
phi_M_100 = read_me(fname)
fname = f"{fdir_300}/phi_M_glial.txt"
phi_M_300 = read_me(fname)

t = np.linspace(0, len(phi_M_100)*1.0e-4, len(phi_M_100))*1000

fname = f"{fdir_100}/Na_ECS_glial.txt"
Na_e_100 = read_me(fname)
fname = f"{fdir_300}/Na_ECS_glial.txt"
Na_e_300 = read_me(fname)

fname = f"{fdir_100}/Na_ICS_glial.txt"
Na_i_100 = read_me(fname)
fname = f"{fdir_300}/Na_ICS_glial.txt"
Na_i_300 = read_me(fname)

fname = f"{fdir_100}/K_ECS_glial.txt"
K_e_100 = read_me(fname)
fname = f"{fdir_300}/K_ECS_glial.txt"
K_e_300 = read_me(fname)

fname = f"{fdir_100}/K_ICS_glial.txt"
K_i_100 = read_me(fname)
fname = f"{fdir_300}/K_ICS_glial.txt"
K_i_300 = read_me(fname)

fname = f"{fdir_100}/i_pump.txt"
i_pump_100 = read_me(fname)
fname = f"{fdir_300}/i_pump.txt"
i_pump_300 = read_me(fname)

fname = f"{fdir_100}/i_kir.txt"
i_kir_100 = read_me(fname)
fname = f"{fdir_300}/i_kir.txt"
i_kir_300 = read_me(fname)

# Concentration plots
fig = plt.figure(figsize=(10,12))
ax = plt.gca()

ax1 = fig.add_subplot(3,2,1)
plt.ylabel(r'[Na]$_e$ (mM)')
plt.plot(t, Na_e_300, linewidth=3, color=red, label="300 Hz")
plt.plot(t, Na_e_100, linewidth=3, color=blue, label="100 Hz")

ax3 = fig.add_subplot(3,2,2)
plt.ylabel(r'[K]$_e$ (mM)')
plt.plot(t, K_e_300, linewidth=3, color=red, label="300 Hz")
plt.plot(t, K_e_100, linewidth=3, color=blue, label="100 Hz")

ax2 = fig.add_subplot(3,2,3)
plt.ylabel(r'[K]$_g$ (mM)')
plt.plot(t, K_i_300, linewidth=3, color=red, label="300 Hz")
plt.plot(t, K_i_100, linewidth=3, color=blue, label="100 Hz")

ax5 = fig.add_subplot(3,2,4)
plt.ylabel(r'$\phi_M^g$ (mV)')
plt.xlabel(r'time (ms)')
plt.plot(t, phi_M_300, linewidth=3, color=red, label="300 Hz")
plt.plot(t, phi_M_100, linewidth=3, color=blue, label="100 Hz")

plt.legend()

ax5 = fig.add_subplot(3,2,5)
plt.ylabel(r'$I_M^g$ ($\rm \mu A/cm^2$) 300 Hz')
plt.xlabel(r'time (ms)')
plt.plot(t, - 2 * np.array(i_pump_300), linewidth=3, color=red, label="pump")
plt.plot(t, i_kir_300, linewidth=3, color=red_dark, label="kir")
plt.plot(t, np.array(i_kir_300) - 2 * np.array(i_pump_300), linestyle='dotted', linewidth=3, color=red_light, label="total")

plt.legend()

ax5 = fig.add_subplot(3,2,6)
plt.ylabel(r'$I_M^g$ ($\rm \mu A/cm^2$) 100 Hz')
plt.xlabel(r'time (ms)')
plt.plot(t, - 2 * np.array(i_pump_100), linewidth=3, color=blue, label="pump")
plt.plot(t, i_kir_100, linewidth=3, color=blue_dark, label="kir")
plt.plot(t, np.array(i_kir_100) - 2 * np.array(i_pump_100), linestyle='dotted', linewidth=3, color=blue_light, label="total")

plt.legend()

# make pretty
ax.axis('off')
plt.tight_layout()

# save figure to file
plt.savefig(f'summary.svg', format='svg')
