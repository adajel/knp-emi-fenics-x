import os
import dolfinx
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import scifem

from knpemi.odeSolver import MembraneModel
from collections import namedtuple

import mm_calibration as ode

M = 10
omega = dolfinx.mesh.create_interval(
    MPI.COMM_WORLD, M, points=(0,1), ghost_mode=dolfinx.mesh.GhostMode.shared_facet
)
tag = 1
cell_map = omega.topology.index_map(omega.topology.dim)
num_cells_local = cell_map.size_local + cell_map.num_ghosts
cell_marker = np.full(num_cells_local, tag, dtype=np.int32)

ct = dolfinx.mesh.meshtags(
    omega, omega.topology.dim, np.arange(num_cells_local, dtype=np.int32), cell_marker
)

Q = dolfinx.fem.functionspace(omega, ("CG", 1))

g_syn_bar = 0
stimulus = {'stim_amplitude': g_syn_bar}

membrane = MembraneModel(ode, ct, tag, Q)

V_index_n = ode.state_indices('V_n')
V_index_g = ode.state_indices('V_g')
K_e_index = ode.state_indices('K_e')
K_n_index = ode.state_indices('K_n')
K_g_index = ode.state_indices('K_g')
Na_e_index = ode.state_indices('Na_e')
Na_n_index = ode.state_indices('Na_n')
Na_g_index = ode.state_indices('Na_g')
Cl_e_index = ode.state_indices('Cl_e')
Cl_n_index = ode.state_indices('Cl_n')
Cl_g_index = ode.state_indices('Cl_g')

n_index = ode.state_indices('n')
m_index = ode.state_indices('m')
h_index = ode.state_indices('h')

potential_history_n = []
potential_history_g = []
K_e_history = []
K_n_history = []
K_g_history = []
Na_e_history = []
Na_n_history = []
Na_g_history = []
Cl_e_history = []
Cl_n_history = []
Cl_g_history = []

n_history = []
m_history = []
h_history = []

for _ in range(10000):
    membrane.step_lsoda(dt=0.1, stimulus=stimulus)

    potential_history_n.append(1*membrane.states[:, V_index_n])
    potential_history_g.append(1*membrane.states[:, V_index_g])
    K_e_history.append(1*membrane.states[:, K_e_index])
    K_n_history.append(1*membrane.states[:, K_n_index])
    K_g_history.append(1*membrane.states[:, K_g_index])
    Na_e_history.append(1*membrane.states[:, Na_e_index])
    Na_n_history.append(1*membrane.states[:, Na_n_index])
    Na_g_history.append(1*membrane.states[:, Na_g_index])
    Cl_e_history.append(1*membrane.states[:, Cl_e_index])
    Cl_n_history.append(1*membrane.states[:, Cl_n_index])
    Cl_g_history.append(1*membrane.states[:, Cl_g_index])

    n_history.append(1*membrane.states[:, n_index])
    m_history.append(1*membrane.states[:, m_index])
    h_history.append(1*membrane.states[:, h_index])

potential_history_n = np.array(potential_history_n)
potential_history_g = np.array(potential_history_g)
K_e_history = np.array(K_e_history)
K_n_history = np.array(K_n_history)
K_g_history = np.array(K_g_history)
Na_e_history = np.array(Na_e_history)
Na_n_history = np.array(Na_n_history)
Na_g_history = np.array(Na_g_history)
Cl_e_history = np.array(Cl_e_history)
Cl_n_history = np.array(Cl_n_history)
Cl_g_history = np.array(Cl_g_history)
n_history = np.array(n_history)
m_history = np.array(m_history)
h_history = np.array(h_history)

print("-------------------------------------------------------------")
print("phi_M_n_init =", potential_history_n[-1, 2])
print("phi_M_g_init =", potential_history_g[-1, 2])
print("K_e_init =", K_e_history[-1, 2])
print("K_n_init =", K_n_history[-1, 2])
print("K_g_init =", K_g_history[-1, 2])
print("Na_e_init =", Na_e_history[-1, 2])
print("Na_n_init =", Na_n_history[-1, 2])
print("Na_g_init =", Na_g_history[-1, 2])
print("Cl_e_init =", Cl_e_history[-1, 2])
print("Cl_n_init =", Cl_n_history[-1, 2])
print("Cl_g_init =", Cl_g_history[-1, 2])
print("n_init =", n_history[-1, 2])
print("m_init =", m_history[-1, 2])
print("h_init =", h_history[-1, 2])
print("-------------------------------------------------------------")

g_leak_Na_g = 0.1      # Na leak conductivity (mS/cm**2)
g_leak_K_g  = 1.696    # K leak conductivity (mS/cm**2)
g_leak_Cl_g = 0.05     # Cl leak conductivity (mS/cm**2)
I_max_g = 10.75975     # max pump strength (muA/cm^2)

m_K = 1.5              # threshold ECS K (mol/m^3)
m_Na = 10              # threshold ICS Na (mol/m^3)
I_max_n = 58.0         # max pump strength (muA/cm^2)

C_M = 1.0              # Faraday's constant (mC/ mol)

# Physical parameters (PDEs)
temperature = 307e3            # temperature (m K)
R = 8.315e3                    # Gas Constant (m J/(K mol))
F = 96500e3                    # Faraday's constant (mC/ mol)

ICS_vol = 3.42e-11/2.0         # ICS volume (cm^3)
ECS_vol = 7.08e-11             # ECS volume (cm^3)
surface = 2.29e-6              # membrane surface (cm²)

K_e_init = 3.092970607490389
K_g_init = 99.3100014897692

# set conductance
phi_M_g = potential_history_g[:, 2]

K_e = K_e_history[:, 2]
K_g = K_g_history[:, 2]

Na_e = Na_e_history[:, 2]
Na_g = Na_g_history[:, 2]

E_K_g = R * temperature / F * np.log(K_e/K_g)
E_K_init = R * temperature / F * np.log(K_e_init/K_g_init)

E_Na_g = R * temperature / F * np.log(Na_e/Na_g)

i_pump = I_max_g \
       * (K_e / (K_e + m_K)) \
       * (Na_g ** (1.5) / (Na_g ** (1.5) + m_Na ** (1.5)))

dphi = phi_M_g - E_K_g
A = 1 + np.exp(18.4/42.4)
B = 1 + np.exp(-(0.1186e3 + E_K_init)/0.0441e3)
C = 1 + np.exp((dphi + 0.0185e3)/0.0425e3)
D = 1 + np.exp(-(0.1186e3 + phi_M_g)/0.0441e3)
g_Kir = np.sqrt(K_e/K_e_init)*(A*B)/(C*D)

# define and return current
I_Kir = g_leak_K_g * g_Kir*(phi_M_g - E_K_g)
I_Na = g_leak_Na_g * (phi_M_g - E_Na_g)

# ODE plots
fig = plt.figure(figsize=(16,12))
ax = plt.gca()

ax1 = fig.add_subplot(3,4,1)
plt.title(r'ECS Na$^+$')
plt.plot(Na_e_history[:, 2], linewidth=3, color='b')

ax2 = fig.add_subplot(3,4,2)
plt.title(r'ECS K$^+$')
plt.plot(K_e_history[:, 2], linewidth=3, color='b')

ax3 = fig.add_subplot(3,4,3)
plt.title(r'Neuron Na$^+$')
plt.plot(Na_n_history[:, 2],linewidth=3, color='r')

ax4 = fig.add_subplot(3,4,4)
plt.title(r'Neuron K$^+$')
plt.plot(K_n_history[:, 2],linewidth=3, color='r')

ax3 = fig.add_subplot(3,4,5)
plt.title(r'Glia Na$^+$')
plt.plot(Na_g_history[:, 2],linewidth=3, color='r')

ax4 = fig.add_subplot(3,4,6)
plt.title(r'Glia K$^+$')
plt.plot(K_g_history[:, 2],linewidth=3, color='r')

ax5 = fig.add_subplot(3,4,7)
plt.title(r'Membrane potential neuron')
plt.plot(potential_history_n[:, 2], linewidth=3)

ax6 = fig.add_subplot(3,4,8)
plt.title(r'Membrane potential glial')
plt.plot(potential_history_g[:, 2], linewidth=3)

ax7 = fig.add_subplot(3,4,9)
plt.title(r'K currents')
plt.plot(I_Kir, linewidth=3)
plt.plot(-2*i_pump, linewidth=3)

ax7 = fig.add_subplot(3,4,10)
plt.title(r'K currents')
plt.plot(I_Na, linewidth=3)
plt.plot(3*i_pump, linewidth=3)

#ax7 = fig.add_subplot(3,4,9)
#plt.title(r'Gating variable n')
#plt.plot(n_history[:, 2], linewidth=3)
#
#ax8 = fig.add_subplot(3,4,10)
#plt.title(r'Gating variable m')
#plt.ylabel(r'$\phi_M$ (mV)')
#plt.plot(m_history[:, 2], linewidth=3)
#
#ax9 = fig.add_subplot(3,4,11)
#plt.title(r'Gating variable h')
#plt.plot(h_history[:, 2], linewidth=3)

# make pretty
ax.axis('off')
plt.tight_layout()

# save figure to file
plt.savefig('calibration.svg', format='svg')
plt.close()
