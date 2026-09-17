import numpy as np
import sys
import os
import argparse
import yaml

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import seaborn as sns

palette = sns.color_palette("husl", 8)
color_palette = [palette.as_hex() for color in palette][1]
colors = [color_palette[7], color_palette[2], color_palette[5]]
linestyles = ['-', '--', ':']

c_D1 = "#6086BF"
c_D2 = "#796BC6"
c_D3 = "#AE69BF"

grey = "#423c3c"
pink = '#e31be3'
blue_light = "#56B4E9"
blue_dark = "#191970"
blue = "#3975db"

colors = [pink, c_D2, grey]

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


def get_normalized_phi_M(phi_M, stimuli_end):

        # Normalized membrane potential over time
        phi_M_max = phi_M[stimuli_end]  # get max value of membrane potential
        phi_M_rest = np.min(phi_M)      # get min value of membrane potential (i.e. the resting potential)

        # calculate normalized membrane potential
        N = len(phi_M)
        phi_M_norm = (phi_M[stimuli_end:] - np.full(N, phi_M_rest)[stimuli_end:])/(phi_M_max - phi_M_rest)

        return phi_M_norm


def plot_traces(config):

    ### ------------------------------------------------------------ ###
    ### Make plot concentrations, potential and normalized potential ###
    ### ------------------------------------------------------------ ###

    alpha_i = 0.11
    alpha_e = 0.22
    gamma_m = 4.33e4 # 1/cm

    lw = 4

    # Create stacked 1D subplots sharing the same x-axis (time)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15*0.85), sharex=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    for i, config in enumerate(configs):

        # Read time parameters from config file
        dt = config['dt']
        save_frequency = config['save_frequency']
        Tstop = config['Tstop']
        t = np.arange(0, Tstop, dt * save_frequency)

        label = config['fname']

        # Get data from results files
        data_dir_name = config['fname']
        fdirs = f"../../results/{data_dir_name}"

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

        ri  = 1 / (sigma_i * alpha_i)   # intracellular resistance k Ohm cm
        re  = 1 / (sigma_e * alpha_e)   # extracellular resistance k Ohm cm
        rm = 1 / (g_tot * gamma_m)      # membrane resistance k Ohm cm**3
        length_constant = np.sqrt(rm / (ri + re)) * 1.0e4 # convert to cm

        # Plot Nernst potential (1D line)
        ax1.plot(t, K_ECS, linewidth=lw, linestyle=linestyles[i], color=colors[i], label=label)
        ax2.plot(t, K_ICS, linewidth=lw, linestyle=linestyles[i], color=colors[i], label=label)
        ax3.plot(t, E_K, linewidth=lw, linestyle=linestyles[i], color=colors[i], label=label)
        ax4.plot(t, I_Kir, linewidth=lw, linestyle=linestyles[i], color=colors[i], label=label)
        ax5.plot(t, phi_M, linewidth=lw, linestyle=linestyles[i], color=colors[i], label=label)
        ax6.plot(t, phi_M, linewidth=lw, linestyle=linestyles[i], color=colors[i], label=label)

        # TODO
        # get index of max value (i.e. where the stimuli is turned off) - same for all
        # model variations
        #stimuli_end = np.argmax(phi_M) + 20
        #print(f"stimuli end: {stimuli_end*0.1*5}")

        #phi_M_norm = get_normalized_phi_M(phi_M, stimuli_end)

        #indices = [i for i, x in enumerate(phi_M_norm) if (x > 0.499 and x < 0.501)]
        #print(indices)
        #print("time constant 3D", indices[0]*dt*save_frequency)
        #t_normalized = np.arange(stimuli_end * dt * save_frequency, Tstop, dt * save_frequency)

        #ax6.plot(t_normalized, phi_M_norm, linewidth=lw, color=pink)

    ax1.set_ylabel(r"$\rm c_{K_e}$ (mM)")   # ECS K concentration
    ax2.set_ylabel(r"$\rm c_{K_i}$ (mM)")   # ICS K concentration
    ax3.set_ylabel(r"$\rm c_{K_e}$ (mM)")   # Nernst potential (1D line)
    ax4.set_ylabel(r"$\rm I_{Kir}$ ($\rm \mu A/cm^2$)") # IKir current
    # Membrane potential
    ax5.set_ylabel(r"$\rm\phi_M$ (mV)")
    ax5.axvline(x=102, color='red', linestyle='--', linewidth=lw*1.2)
    ax5.set_xlabel(r"time (ms)")
    # Normalized membrane potential
    #ax6.plot([stimuli_end * dt * save_frequency, Tstop], [0.5, 0.5], color='grey', linestyle="dotted", linewidth=lw*1.2)
    #ax6.set_ylabel(r"normalized $\rm\phi_M$")
    #ax6.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax6.set_xlabel(r"time (ms)")

    # Make pretty and save
    plt.tight_layout()
    plt.legend()
    os.makedirs('results', exist_ok=True)
    fig.savefig('results/3D.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        metavar="config.yml",
        help="path(s) to config file",
        type=str,
        nargs="+",
    )
    conf_arg = vars(parser.parse_args())
    config_file_paths = conf_arg["c"]

    configs = []
    for config_file_path in config_file_paths:
        with open(f"../../config_files/{config_file_path}.yml") as conf_file:
            configs.append(yaml.load(conf_file, Loader=yaml.FullLoader))

    # Pass the list of configs to plotter function
    plot_traces(configs)
