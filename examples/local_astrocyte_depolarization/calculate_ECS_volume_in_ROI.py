#!/usr/bin/python3
from knpemi.emiWeakForm import emi_system, create_functions_emi
from knpemi.knpWeakForm import knp_system, create_functions_knp

from knpemi.pdeSolver import create_solver_emi
from knpemi.pdeSolver import create_solver_knp

from knpemi.utils import set_initial_conditions, setup_membrane_model
from knpemi.utils import interpolate_to_membrane
from knpemi.utils import update_ode_variables
from knpemi.utils import update_pde_variables

import mm_glial as mm_glial
import mm_hh as mm_hh

import dolfinx
import scifem
from mpi4py import MPI
import numpy as np
import argparse
import yaml
import dolfinx

import ufl

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
    """ Solve system (PDEs and ODEs) """

    x_L = config['x_L']; x_U = config['x_U'];
    y_L = config['x_L']; y_U = config['x_U'];
    z_L = config['x_L']; z_U = config['x_U'];

    mesh_file = config['mesh_file'] # path to mesh file
    fname = config["fname"]         # directory for saving results

    print(f'{bcolors.OKBLUE}Reading mesh from {mesh_file} ...')
    mesh, ct, ft = read_mesh(mesh_file)
    print(f'mesh read. ms{bcolors.ENDC}')

    # Spatial coordinates
    x, y, z = SpatialCoordinate(mesh)

    # The region of interest is defined by x_U, x_L, y_U, y_L, z_U, z_L)
    box_condition = And(gt(x, x_L),
                    And(lt(x, x_U),
                    And(lt(y, y_U),
                    And(gt(y, y_L),
                    And(gt(z, z_L), lt(z, z_U))))))

    # Convert boolean condition into numerical mask (1.0 inside, 0.0 outside)
    roi_indicator = conditional(box_condition, 1.0, 0.0)

    # Integrate 1.0 over the tagged subdomain and box ROI
    dx = Measure("dx", domain=mesh, subdomain_data=ct)
    volume_form = dolfinx.fem.form(roi_indicator * dx(0))

    local_volume = dolfinx.fem.assemble_scalar(volume_form)
    global_volume = mesh.comm.allreduce(local_volume, op=MPI.SUM)*1.0e12
    print(f"Volume of subdomain ECS inside ROI: {global_volume} um^3")

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

    with open(f"config_files/{config_file_path}.yml") as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)

    calculate_volume_ECS(config)
