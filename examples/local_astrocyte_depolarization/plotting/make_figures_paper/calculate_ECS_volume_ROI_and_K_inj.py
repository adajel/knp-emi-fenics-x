#!/usr/bin/python3
import dolfinx
from mpi4py import MPI
import numpy as np
import argparse
import yaml
import dolfinx

from ufl import (
        ln,
        SpatialCoordinate,
        conditional,
        Measure,
        And,
        lt,
        le,
        gt,
        ge,
)

# Define colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

i_res = "-"
e_res = "+"

comm = MPI.COMM_WORLD

def read_mesh(mesh_file):

    # Set ghost mode
    ghost_mode = dolfinx.mesh.GhostMode.shared_facet

    with dolfinx.io.XDMFFile(comm, mesh_file, 'r') as xdmf:
        # Read mesh and cell tags
        mesh = xdmf.read_mesh(ghost_mode=ghost_mode, name="Grid")
        ct = xdmf.read_meshtags(mesh, name='cell_marker')

        # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
        mesh.topology.create_entities(mesh.topology.dim-1)
        mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

        # Read facets
        ft = xdmf.read_meshtags(mesh, name='facet_marker')

    xdmf.close()

    return mesh, ct, ft


def calculate_volume_ECS(config):
    """ Calculate the ECS volume inside ROI """

    x_L = config['x_L']; x_U = config['x_U'];
    y_L = config['y_L']; y_U = config['y_U'];
    z_L = config['z_L']; z_U = config['z_U'];

    mesh_file = config['mesh_file'] # path to mesh file
    fname = config["fname"]         # directory for saving results

    print(f'{bcolors.OKBLUE}Reading mesh from {mesh_file} ...')
    mesh, ct, ft = read_mesh(f"../../{mesh_file}")
    print(f'mesh read. ms{bcolors.ENDC}')

    # Spatial coordinates
    x, y, z = SpatialCoordinate(mesh)

    # The region of interest is defined by x_U, x_L, y_U, y_L, z_U, z_L)
    in_box = And(gt(x, x_L),
             And(lt(x, x_U),
             And(lt(y, y_U),
             And(gt(y, y_L),
             And(gt(z, z_L), lt(z, z_U))))))

    # Convert boolean condition into numerical mask (1.0 inside, 0.0 outside)
    roi_indicator = conditional(in_box, 1.0, 0.0)

    # Integrate 1.0 over the tagged subdomain and box ROI
    dx = Measure("dx", domain=mesh, subdomain_data=ct)

    form_ECS = dolfinx.fem.form(roi_indicator * dx(0))
    form_glia = dolfinx.fem.form(roi_indicator * dx(2))
    form_neuro = dolfinx.fem.form(roi_indicator * dx(1))

    vol_ECS = dolfinx.fem.assemble_scalar(form_ECS)*1.0e12
    vol_glia = dolfinx.fem.assemble_scalar(form_glia)*1.0e12
    vol_neuro = dolfinx.fem.assemble_scalar(form_neuro)*1.0e12
    vol_tot = vol_ECS + vol_glia + vol_neuro

    print(f"ECS volume in ROI: {vol_ECS} um^3")
    print(f"Total volume of ROI: {vol_tot} um^3")

    # Strength of source term
    f_value = config["f_value"]
    # Frequency of source term (application of source term)
    period = config["period"]           # repeat every period (frequency)
    pulse_width = config["pulse_width"] # duration (ms)
    delay = config["delay"]             # start offset (ms)
    end_time = config["end_time"]       # turn source term off after end_time (ms)

    # Time variables
    t = dolfinx.fem.Constant(mesh, 0.0)
    Tstop = config["Tstop"]
    dt = 0.1

    # NB! As modulo is not supported by UFL, the source term is defined as a
    # constant, and updated in the time-loop further down. If the
    # source term is changed, the time loop further down must also be updated.
    # To be fixed..
    source_active = dolfinx.fem.Constant(mesh, 0.0)
    source_active.value = 1 if (t.value - delay) % period < pulse_width else 0

    # Define when (t) and where (x, y, z) source term is applied. The source
    # terms is on for 1 ms every 10th ms with a delay of 0.2 ms, i.e. the pulse
    # is on if: (t >= delay) and ((t - delay) % period < pulse_width). The
    # source  term is applied in a region of interest defined by x_U, x_L, y_U,
    # y_L, z_U, z_L)
    f_condition = And(ge(t, delay),
                  And(le(t, end_time),
                  And(gt(x, x_L),
                  And(lt(x, x_U),
                  And(lt(y, y_U),
                  And(gt(y, y_L),
                  And(gt(z, z_L), lt(z, z_U))))))))

    # Define source term
    f_source_K = conditional(f_condition, f_value, 0) * source_active
    form_source = dolfinx.fem.form(f_source_K * dx(0))

    K_injected = 0

    for k in range(int(round(Tstop/float(dt)))):
        # add contribution from source term
        K_injected += dolfinx.fem.assemble_scalar(form_source)*dt

        # Update time and source terms
        t.value = float(t + dt)
        source_active.value = 1 if (t.value - delay) % period < pulse_width else 0

    print(f"Total K+ injection in ROI: {K_injected} mM")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        metavar="config.yml",
        help="path to config file",
        type=str,
    )
    conf_arg = vars(parser.parse_args())
    config_file_path = conf_arg["c"]

    with open(f"../../config_files/{config_file_path}.yml") as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)

    calculate_volume_ECS(config)
