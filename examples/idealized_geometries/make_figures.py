import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import numpy as np
import os

import dolfinx
import adios4dolfinx
from mpi4py import MPI
import scifem

# Set font & text parameters
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 13}

plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'

comm = MPI.COMM_WORLD

def get_time_series_sub(checkpoint_fname, point, tag, dt, Tstop):

    # Read mesh
    mesh_sub = adios4dolfinx.read_mesh(checkpoint_fname, MPI.COMM_WORLD)

    # Create function space and functions for storing data
    V = dolfinx.fem.functionspace(mesh_sub, ("CG", 1))
    K = dolfinx.fem.Function(V)
    Cl = dolfinx.fem.Function(V)
    Na = dolfinx.fem.Function(V)
    phi = dolfinx.fem.Function(V)

    # Create list for point evaluation of functions over time
    Nas = []
    Ks = []
    Cls = []
    phis = []

    t = dt
    while t <= Tstop:
        adios4dolfinx.read_function(checkpoint_fname, K, time=t, name=f"c_K_{tag}")
        adios4dolfinx.read_function(checkpoint_fname, Cl, time=t, name=f"c_Cl_{tag}")
        adios4dolfinx.read_function(checkpoint_fname, Na, time=t, name=f"c_Na_{tag}")
        adios4dolfinx.read_function(checkpoint_fname, phi, time=t, name=f"phi_{tag}")

        # K concentrations
        Ks.append(scifem.evaluate_function(K, point)[0][0])
        Cls.append(scifem.evaluate_function(Cl, point)[0][0])
        Nas.append(scifem.evaluate_function(Na, point)[0][0])
        phis.append(scifem.evaluate_function(phi, point)[0][0])

        t += dt
        print("t=", t)

    return Nas, Ks, Cls, phis

def get_time_series_mem(checkpoint_fname, point, tag, dt, Tstop):

    # Read mesh
    mesh_mem = adios4dolfinx.read_mesh(checkpoint_fname, MPI.COMM_WORLD)

    # Create function space and function for storing data
    V = dolfinx.fem.functionspace(mesh_mem, ("CG", 1))
    phi_M = dolfinx.fem.Function(V)

    tr_Ki = dolfinx.fem.Function(V)
    tr_Ke = dolfinx.fem.Function(V)
    tr_Nai = dolfinx.fem.Function(V)
    tr_Nae = dolfinx.fem.Function(V)
    tr_Cli = dolfinx.fem.Function(V)
    tr_Cle = dolfinx.fem.Function(V)

    # Create list for point evaluation of functions over time
    phi_Ms = []
    tr_K_es = []
    tr_K_is = []
    tr_Cl_es = []
    tr_Cl_is = []
    tr_Na_es = []
    tr_Na_is = []

    t = dt
    while t <= Tstop:
        # Membrane potential
        adios4dolfinx.read_function(checkpoint_fname, phi_M, time=t,name=f"phi_M_{tag}")
        phi_Ms.append(scifem.evaluate_function(phi_M, point)[0][0])

        # Trace of K from ICS
        adios4dolfinx.read_function(checkpoint_fname, tr_Ki, time=t, name=f"c_K_{tag}")
        tr_K_is.append(scifem.evaluate_function(tr_Ki, point)[0][0])

        # Trace of Cl from ICS
        adios4dolfinx.read_function(checkpoint_fname, tr_Cli, time=t, name=f"c_Cl_{tag}")
        tr_Cl_is.append(scifem.evaluate_function(tr_Cli, point)[0][0])

        # Trace of Na from ICS
        adios4dolfinx.read_function(checkpoint_fname, tr_Nai, time=t, name=f"c_Na_{tag}")
        tr_Na_is.append(scifem.evaluate_function(tr_Nai, point)[0][0])

        # Trace of K from ECS
        adios4dolfinx.read_function(checkpoint_fname, tr_Ke, time=t, name=f"c_K_{0}")
        tr_K_es.append(scifem.evaluate_function(tr_Ke, point)[0][0])

        # Trace of Cl from ECS
        adios4dolfinx.read_function(checkpoint_fname, tr_Cle, time=t, name=f"c_Cl_{0}")
        tr_Cl_es.append(scifem.evaluate_function(tr_Cle, point)[0][0])

        # Trace of Na from ECS
        adios4dolfinx.read_function(checkpoint_fname, tr_Nae, time=t, name=f"c_Na_{0}")
        tr_Na_es.append(scifem.evaluate_function(tr_Nae, point)[0][0])

        t += dt
        print("t=", t)

    return phi_Ms, tr_K_es, tr_K_is, tr_Na_es, tr_Na_is, tr_Cl_es, tr_Cl_is

def plot_concentration(fname, dt, Tstop, points):

    temperature = 300 # temperature (K)
    F = 96485         # Faraday's constant (C/mol)
    R = 8.314         # Gas constant (J/(K*mol))

    time = 1.0e3*np.arange(0, T-dt, dt)

    point_i = points['ICS']
    point_e = points['ECS']
    point_m = points['mem']

    #################################################################
    # get data axon A is stimulated
    checkpoint_fname_e = f'results/{fname}/checkpoint_sub_0.bp'
    checkpoint_fname_i = f'results/{fname}/checkpoint_sub_1.bp'
    checkpoint_fname_M = f'results/{fname}/checkpoint_mem_1.bp'

    # bulk concentrations
    tag_e = 0
    tag_i = 1

    Na_e, K_e, Cl_e, phi_e = get_time_series_sub(checkpoint_fname_e, point_e, tag_e, dt, Tstop)
    Na_i, K_i, Cl_i, phi_i = get_time_series_sub(checkpoint_fname_i, point_i, tag_i, dt, Tstop)
    phi_M, tr_K_e, tr_K_i, tr_Na_e, tr_Na_i, tr_Cl_e, tr_Cl_i = get_time_series_mem(checkpoint_fname_M, point_M, tag_i, dt, Tstop)

    temperature = 300e3; F = 96485e3; R = 8.314e3
    # Calculate Nernst potentials
    E_Na = R * temperature / (F) * np.log(np.array(tr_Na_e) / np.array(tr_Na_i))
    E_K = R * temperature / (F) * np.log(np.array(tr_K_e) / np.array(tr_K_i))

    #################################################################
    # get data axons BC are stimulated

    # Concentration plots
    fig = plt.figure(figsize=(11, 11))
    ax = plt.gca()

    ax1 = fig.add_subplot(3,3,1)
    plt.title(r'Na$^+$ concentration (ECS)')
    plt.ylabel(r'[Na]$_e$ (mM)')
    plt.plot(Na_e, linewidth=3, color='b')

    ax2 = fig.add_subplot(3,3,2)
    plt.title(r'K$^+$ concentration (ECS)')
    plt.ylabel(r'[K]$_e$ (mM)')
    plt.plot(K_e, linewidth=3, color='b')

    ax3 = fig.add_subplot(3,3,3)
    plt.title(r'Cl$^-$ concentration (ECS)')
    plt.ylabel(r'[Cl]$_e$ (mM)')
    plt.plot(Cl_e, linewidth=3, color='b')

    ax4 = fig.add_subplot(3,3,4)
    plt.title(r'Na$^+$ concentration (ICS)')
    plt.ylabel(r'[Na]$_i$ (mM)')
    plt.plot(Na_i,linewidth=3, color='r')

    ax5 = fig.add_subplot(3,3,5)
    plt.title(r'K$^+$ concentration (ICS)')
    plt.ylabel(r'[K]$_i$ (mM)')
    plt.plot(K_i,linewidth=3, color='r')

    ax6 = fig.add_subplot(3,3,6)
    plt.title(r'Cl$^-$ concentration (ICS)')
    plt.ylabel(r'[Cl]$_i$ (mM)')
    plt.plot(Cl_i,linewidth=3, color='r')

    ax7 = fig.add_subplot(3,3,7)
    plt.title(r'Membrane potential')
    plt.ylabel(r'$\phi_M$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(phi_M, linewidth=3)

    ax8 = fig.add_subplot(3,3,8)
    plt.title(r'Nernst potential K$^+$')
    plt.ylabel(r'$\rm E_{K^+}$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(E_K, linewidth=3)

    ax9 = fig.add_subplot(3,3,9)
    plt.title(r'Nernst potential Na$^+$')
    plt.ylabel(r'$\rm E_{Na^+}$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(E_Na, linewidth=3)

    # make pretty
    ax.axis('off')
    plt.tight_layout()

    # save figure to file
    plt.savefig(f'results/{fname}/summary.svg', format='svg')

    f_phi_M = open(f'results/{fname}/phi_M.txt', "w")
    for p in phi_M:
        f_phi_M.write("%.10f \n" % p)
    f_phi_M.close()

    f_K_e = open(f'results/{fname}/K_ECS.txt', "w")
    for p in K_e:
        f_K_e.write("%.10f \n" % p)
    f_K_e.close()

    f_K_i = open(f'results/{fname}/K_ICS.txt', "w")
    for p in K_i:
        f_K_i.write("%.10f \n" % p)
    f_K_i.close()

    f_Na_e = open(f'results/{fname}/Na_ECS.txt', "w")
    for p in Na_e:
        f_Na_e.write("%.10f \n" % p)
    f_Na_e.close()

    f_Na_i = open(f'results/{fname}/Na_ICS.txt', "w")
    for p in Na_i:
        f_Na_i.write("%.10f \n" % p)
    f_Na_i.close()

    return

# Time variables
dt = 1.0e-4
T = 1.0e-1

# Make 2D plot
fname = "2D"
# create directory for figures
if not os.path.isdir(f'results/{fname}'):
    os.mkdir(f'results/{fname}')

# Point to evaluate solution in 2D geometry
x_M = 25; y_M = 3
x_e = 25; y_e = 3.5
x_i = 25; y_i = 2
point_M = np.array([[x_M * 1.0e-6, y_M * 1.0e-6]])
point_e = np.array([[x_e * 1.0e-6, y_e * 1.0e-6]])
point_i = np.array([[x_i * 1.0e-6, y_i * 1.0e-6]])
points = {'ECS':point_e, 'ICS': point_i, 'mem':point_M}
plot_concentration(fname, dt, T, points)

# Make 3D plot
fname = "3D"
# create directory for figures
if not os.path.isdir(f'results/{fname}'):
    os.mkdir(f'results/{fname}')

# Point to evaluate solution in 3D geometry
x_M = 25.6; y_M = 0.34; z_M = 0.4
x_e = 25; y_e = 0.45; z_e = 0.65
x_i = 25; y_i = 0.3; z_i = 0.3
point_M = np.array([[x_M * 1.0e-6, y_M * 1.0e-6, z_M * 1.0e-6]])
point_e = np.array([[x_e * 1.0e-6, y_e * 1.0e-6, z_e * 1.0e-6]])
point_i = np.array([[x_i * 1.0e-6, y_i * 1.0e-6, z_i * 1.0e-6]])
points = {'ECS':point_e, 'ICS': point_i, 'mem':point_M}
plot_concentration(fname, dt, T, points)
