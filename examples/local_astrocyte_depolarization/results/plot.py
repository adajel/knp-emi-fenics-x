import numpy as np
import sys

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

blue1 = '#365C8D'
orange = '#FFA500'
green1 = '#A0DA39'

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
fdir_300 = "local_PAP_depolarization_300_hz"

# write phi_M
fname = f"{fdir_100}/phi_M.txt"
phi_M_100 = read_me(fname)
fname = f"{fdir_300}/phi_M.txt"
phi_M_300 = read_me(fname)

t = np.linspace(0, len(phi_M_100)*1.0e-4, len(phi_M_100))*1000

fname = f"{fdir_100}/Na_ECS.txt"
Na_e_100 = read_me(fname)
fname = f"{fdir_300}/Na_ECS.txt"
Na_e_300 = read_me(fname)

fname = f"{fdir_100}/Na_ICS.txt"
Na_i_100 = read_me(fname)
fname = f"{fdir_300}/Na_ICS.txt"
Na_i_300 = read_me(fname)

fname = f"{fdir_100}/K_ECS.txt"
K_e_100 = read_me(fname)
fname = f"{fdir_300}/K_ECS.txt"
K_e_300 = read_me(fname)

fname_ = f"{fdir_100}/K_ICS.txt"
K_i_100 = read_me(fname)
fname = f"{fdir_300}/K_ICS.txt"
K_i_300 = read_me(fname)

# Concentration plots
fig = plt.figure(figsize=(10*0.9,12*0.9))
ax = plt.gca()

ax1 = fig.add_subplot(3,2,1)
plt.title(r'Na$^+$ concentration (ECS)')
plt.ylabel(r'[Na]$_e$ (mM)')
plt.plot(Na_e_100, linewidth=3, color=orange, label="100 Hz")
plt.plot(Na_e_300, linewidth=3, color=blue1, label="300 Hz")

ax3 = fig.add_subplot(3,2,2)
plt.title(r'K$^+$ concentration (ECS)')
plt.ylabel(r'[K]$_e$ (mM)')
plt.plot(K_e_100, linewidth=3, color=orange, label="100 Hz")
plt.plot(K_e_300, linewidth=3, color=blue1, label="300 Hz")

ax2 = fig.add_subplot(3,2,3)
plt.title(r'Na$^+$ concentration (ICS)')
plt.ylabel(r'[Na]$_i$ (mM)')
plt.plot(Na_i_100,linewidth=3, color=orange, label="100 Hz")
plt.plot(Na_i_300,linewidth=3, color=blue1, label="300 Hz")

ax2 = fig.add_subplot(3,2,4)
plt.title(r'K$^+$ concentration (ICS)')
plt.ylabel(r'[K]$_i$ (mM)')
plt.plot(K_i_100, linewidth=3, color=orange, label="100 Hz")
plt.plot(K_i_300, linewidth=3, color=blue1, label="300 Hz")

ax5 = fig.add_subplot(3,2,5)
plt.title(r'Membrane potential')
plt.ylabel(r'$\phi_M$ (mV)')
plt.xlabel(r'time (ms)')
plt.plot(phi_M_100, linewidth=3, color=orange, label="100 Hz")
plt.plot(phi_M_300, linewidth=3, color=blue1, label="300 Hz")

plt.legend()

# make pretty
ax.axis('off')
plt.tight_layout()

# save figure to file
plt.savefig(f'summary.svg', format='svg')
