from pathlib import Path
import pyvista
import seaborn
import argparse
import yaml

import numpy as np
import dolfinx
import adios4dolfinx.backends.xdmf.backend
from mpi4py import MPI

# Allow to plot empty meshes
pyvista.global_theme.allow_empty_mesh = True

c_point = "#00FFFF"

def get_grid_field(finame, funame, time_index):
    # Read mesh from file
    filename = f"../../results/{finame}.xdmf"
    function_info = adios4dolfinx.backends.xdmf.backend.extract_function_names_and_timesteps(filename)
    grid = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="xdmf")

    # Assert that funame is function name
    function_names = adios4dolfinx.read_function_names(filename, MPI.COMM_WORLD,
            backend="xdmf", backend_args={})
    assert f"{funame}" in function_names

    # Assert that time is timestamp
    timestamps = adios4dolfinx.read_timestamps(filename, MPI.COMM_WORLD,
            funame,
            backend="xdmf", backend_args={})

    # Get time based on provided index
    time = float(timestamps[time_index])
    print(f"time: {time}")

    float_stamps = np.array(timestamps, dtype=np.float64)
    pos = np.flatnonzero(np.isclose(float_stamps, time))
    assert len(pos) == 1

    # Read data from file
    p0 = adios4dolfinx.read_point_data(
        filename, f"{funame}", grid, timestamps[pos[0]], backend="xdmf")
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(p0.function_space))
    grid.point_data[f"{funame}"] = p0.x.array

    return grid

def plot_ECS_concentration(fname, grid_ECS, custom_labels, cmap, clim):

    roi_point = pyvista.PolyData([2.5, 2.5, 2.5])
    clip_ECS = grid_ECS.clip()

    position_bar=[0.25, 0.83]
    position_x = position_bar[0]
    position_y = position_bar[1]

    sargs = dict(title="",
                 vertical=True,
                 position_x=position_x,
                 position_y=position_y,
                 height=0.5,
                 width=0.1,
                 n_labels=0,
                 label_font_size=27,
    )

    # Plot ECS concentration
    p = pyvista.Plotter(off_screen=True)

    # Add ECS concentration
    p.add_mesh(clip_ECS,
               cmap=cmap,
               scalar_bar_args=sargs,
               annotations=custom_labels,
               clim=clim,
               show_scalar_bar=False,
    )

    # Add membrane point in ROI
    p.add_mesh(roi_point,
               color=c_point,
               point_size=25,
               render_points_as_spheres=True
    )

    # Make pretty and save
    p.reset_camera()
    p.camera.zoom(1.0)
    p.camera_position = "yz"
    p.screenshot(f"{fname}.png", transparent_background=True)
    p.close()

    return


def plot_ECS_colorbar(fname, custom_labels, cmap, clim):
    """
    Renders and exports a standalone colorbar for the ECS concentration plot.
    """
    # 1. Setup a dedicated canvas size for the colorbar
    p = pyvista.Plotter(window_size=[200, 500], off_screen=True)

    # 2. Configure scalar bar arguments (centered layout)
    sargs = dict(
        title="",
        n_labels=0,
        vertical=True,
        position_x=0.4,
        position_y=0.1,
        width=0.7,
        height=0.85,
        label_font_size=40,
    )

    # 3. Create a dummy PolyData mesh to bind the colorbar properties
    dummy_mesh = pyvista.PolyData([0.0, 0.0, 0.0])
    dummy_mesh.point_data["uh"] = np.array([clim[0]])

    p.add_mesh(
        dummy_mesh,
        scalars="uh",
        cmap=cmap,
        clim=clim,
        annotations=custom_labels,
        scalar_bar_args=sargs,
        show_scalar_bar=True
    )

    # 4. Add the ion concentration title matching plot_ECS_concentration
    p.add_text(
        r"$c_e$ (mM)",
        position=(0.75, 0.65),
        font_size=20,
        viewport=True,
        orientation=-90,
    )

    # 5. Render and export with a transparent background
    p.screenshot(f"{fname}.png", transparent_background=True)
    p.close()

    return

if __name__ == "__main__":

    # Create directory for plots if it doesn't exist
    output_dir = Path(f"results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set color map etc. ECS concentration
    cmap_ECS_K = seaborn.color_palette("BuPu", as_cmap=True)
    clim_ECS_K = [0.0, 4.5]
    custom_labels_ECS_K = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
    fname_ec = f"{output_dir}/ECS_diffusion_colorbar"

    plot_ECS_colorbar(fname_ec, custom_labels_ECS_K, cmap_ECS_K, clim_ECS_K)

    i = 1
    for time_index in [0, 15]:

        # Get solution ECS K+ concentration at time time_index and time 0
        grid_ECS = get_grid_field("ECS_diffusion", "uh", time_index)

        fname_ECS = f"results/ECS_diffusion_{i}"
        plot_ECS_concentration(fname_ECS, grid_ECS,
                custom_labels_ECS_K, cmap_ECS_K, clim_ECS_K)

        i += 1
