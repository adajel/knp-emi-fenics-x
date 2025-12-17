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
        # Read results from file
        adios4dolfinx.read_function(checkpoint_fname, K, time=t, name=f"c_K_{tag}")
        adios4dolfinx.read_function(checkpoint_fname, Cl, time=t, name=f"c_Cl_{tag}")
        adios4dolfinx.read_function(checkpoint_fname, Na, time=t, name=f"c_Na_{tag}")
        adios4dolfinx.read_function(checkpoint_fname, phi, time=t, name=f"phi_{tag}")

        # Append (results) function evaluated in point to list
        Ks.append(scifem.evaluate_function(K, point)[0])
        Cls.append(scifem.evaluate_function(Cl, point)[0])
        Nas.append(scifem.evaluate_function(Na, point)[0])
        phis.append(scifem.evaluate_function(phi, point)[0])

        t += dt

    return Nas, Ks, Cls, phis

def get_time_series_mem(checkpoint_fname, point, tag, dt, Tstop):
    # Read mesh
    mesh_mem = adios4dolfinx.read_mesh(checkpoint_fname, MPI.COMM_WORLD)
    # Create function space and function for storing data
    V = dolfinx.fem.functionspace(mesh_mem, ("CG", 1))
    phi_M = dolfinx.fem.Function(V)

    # Create list for point evaluation of functions over time
    phi_Ms = []

    t = dt
    while t <= Tstop:
        print(t)
        adios4dolfinx.read_function(checkpoint_fname, phi_M, time=t, name=f"phi_M_{tag}")
        print(phi_M.x.array[:])
        print(scifem.evaluate_function(phi_M, point)[0])
        phi_Ms.append(scifem.evaluate_function(phi_M, point)[0])
        t += dt

    return phi_Ms

def plot_3D_concentration(dt, Tstop):

    temperature = 300e3 # temperature (K)
    F = 96485e3         # Faraday's constant (C/mol)
    R = 8.314e3         # Gas constant (J/(K*mol))

    time = np.arange(0, T-dt, dt)

    x_M = 0.0002491287275961443
    y_M = 0.00024278379996648452
    z_M = 0.00023517415844465526

    x_e = 0.0002479871894748248
    y_e = 0.0002424216086463334
    z_e = 0.00023859662336311367

    x_i = 0.00024270178661651962
    y_i = 0.0002496048874198617
    z_i = 0.00022946383703513403

    point_e = np.array([[x_e, y_e, z_e]])
    point_i = np.array([[x_i, y_i, z_i]])
    point_M = np.array([[x_M, y_M, z_M]])

    #################################################################
    # get data axon A is stimulated
    checkpoint_fname_e = 'results/3D/checkpoint_sub_0.bp'
    checkpoint_fname_i = 'results/3D/checkpoint_sub_1.bp'
    checkpoint_fname_M = 'results/3D/checkpoint_mem_1.bp'

    # bulk concentrations
    tag_e = 0
    tag_i = 1

    Na_e, K_e, Cl_e, phi_e = get_time_series_sub(checkpoint_fname_e, point_e, tag_e, dt, Tstop)
    Na_i, K_i, Cl_i, phi_i = get_time_series_sub(checkpoint_fname_i, point_i, tag_i, dt, Tstop)
    phi_M = get_time_series_mem(checkpoint_fname_M, point_M, tag_i, dt, Tstop)

    #################################################################
    # get data axons BC are stimulated

    # Concentration plots
    fig = plt.figure(figsize=(12*0.9,12*0.9))
    ax = plt.gca()

    ax1 = fig.add_subplot(3,3,1)
    plt.title(r'Na$^+$ concentration (ECS)')
    plt.ylabel(r'[Na]$_e$ (mM)')
    plt.plot(Na_e, linewidth=3, color='b')

    ax3 = fig.add_subplot(3,3,2)
    plt.title(r'K$^+$ concentration (ECS)')
    plt.ylabel(r'[K]$_e$ (mM)')
    plt.plot(K_e, linewidth=3, color='b')

    ax3 = fig.add_subplot(3,3,3)
    plt.title(r'Cl$^-$ concentration (ECS)')
    plt.ylabel(r'[Cl]$_e$ (mM)')
    plt.plot(Cl_e, linewidth=3, color='b')

    ax2 = fig.add_subplot(3,3,4)
    plt.title(r'Na$^+$ concentration (ICS)')
    plt.ylabel(r'[Na]$_i$ (mM)')
    plt.plot(Na_i,linewidth=3, color='r')

    ax2 = fig.add_subplot(3,3,5)
    plt.title(r'K$^+$ concentration (ICS)')
    plt.ylabel(r'[K]$_i$ (mM)')
    plt.plot(K_i,linewidth=3, color='r')

    ax2 = fig.add_subplot(3,3,6)
    plt.title(r'Cl$^-$ concentration (ICS)')
    plt.ylabel(r'[Cl]$_i$ (mM)')
    plt.plot(Cl_i,linewidth=3, color='r')

    ax5 = fig.add_subplot(3,3,7)
    plt.title(r'Membrane potential')
    plt.ylabel(r'$\phi_M$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(phi_M, linewidth=3)

    # make pretty
    ax.axis('off')
    plt.tight_layout()

    # save figure to file
    plt.savefig('results/3D/pot_con_3D.svg', format='svg')

    f_phi_M = open('results/3D/phi_M_2D.txt', "w")
    for p in phi_M:
        f_phi_M.write("%.10f \n" % p)
    f_phi_M.close()

    f_K_e = open('results/3D/K_ECS_2D.txt', "w")
    for p in K_e:
        f_K_e.write("%.10f \n" % p)
    f_K_e.close()

    f_K_i = open('results/3D/K_ICS_2D.txt', "w")
    for p in K_i:
        f_K_i.write("%.10f \n" % p)
    f_K_i.close()

    f_Na_e = open('results/3D/Na_ECS_2D.txt', "w")
    for p in Na_e:
        f_Na_e.write("%.10f \n" % p)
    f_Na_e.close()

    f_Na_i = open('results/3D/Na_ICS_2D.txt', "w")
    for p in Na_i:
        f_Na_i.write("%.10f \n" % p)
    f_Na_i.close()

    return

# create directory for figures
if not os.path.isdir('results/3D'):
    os.mkdir('results/3D')

# create figures
dt = 0.1
T = 10

plot_3D_concentration(dt, T)
