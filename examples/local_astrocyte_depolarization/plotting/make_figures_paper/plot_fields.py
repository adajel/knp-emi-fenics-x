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

def get_vw_average(mesh, ARRAY_NAME):

    # 1. Convert points to cells if necessary
    mesh = mesh.point_data_to_cell_data()

    # 2. Compute geometry sizes
    mesh = mesh.compute_cell_sizes()

    field_data = mesh.cell_data[ARRAY_NAME]

    # 3. Dynamic Fallback: Check if it's a 3D volume or a 2D surface area
    if "Volume" in mesh.cell_data and np.sum(np.abs(mesh.cell_data["Volume"])) > 0:
        # It's a 3D Solid mesh. Take absolute values to fix inverted cells!
        sizes = np.abs(mesh.cell_data["Volume"])
        size_type = "Volume-Weighted"
    elif "Area" in mesh.cell_data and np.sum(np.abs(mesh.cell_data["Area"])) > 0:
        # It's a 2D Surface/Shell mesh.
        sizes = np.abs(mesh.cell_data["Area"])
        size_type = "Area-Weighted"

    # 4. Filter out any cells that have exactly 0 size to ensure no division by zero
    valid_idx = sizes > 0
    filtered_data = field_data[valid_idx]
    filtered_sizes = sizes[valid_idx]

    total_size = np.sum(filtered_sizes)

    # 5. Compute the weighted average safely
    if filtered_data.ndim > 1:
        weighted_average = np.sum(filtered_data * filtered_sizes[:, None], axis=0) / total_size
    else:
        weighted_average = np.sum(filtered_data * filtered_sizes) / total_size

    print(f"{size_type} Average of {ARRAY_NAME}: {weighted_average}")

    return weighted_average


def get_grid_field(dir, finame, funame, time_index):
    # Read mesh from file
    filename = f"../../results/{dir}/{finame}.xdmf"
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

def plot_glial_potential(fname, roi_box, roi_bounds, roi_point, ri_grid_glial, \
                         ri_grid_glial_init, clim, custom_labels, camera_position):

    # Assign difference from baseline back to the mesh
    diff_array = grid_glial.point_data["phi_M_2"] - grid_glial_init.point_data["phi_M_2"]
    grid_glial["diff"] = diff_array

    position_bar=[0.9, 0.27]
    position_text=(0.95, 0.60)
    position_x = position_bar[0]
    position_y = position_bar[1]

    sargs = dict(title="",
                 n_labels=0,                # Number of labels
                 vertical=True,             # Horizontal orientation
                 position_x=position_x,     # Move left/right (0 to 1)
                 position_y=position_y,     # Move up/down (0 to 1)
                 width=0.1,                 # Width of the bar
                 height=0.5,                # Height of the bar
                 label_font_size=27,
    )

    # Plot glial membrane potential
    p = pyvista.Plotter(off_screen=True)

    # Add glial membrane potential
    p.add_mesh(grid_glial,
              scalars="diff",
              scalar_bar_args=sargs,
              cmap=cmap_glial,
              clim=clim,
              annotations=custom_labels,
    )
    # Add ROI box
    p.add_mesh(roi_box,
               color="black",
               style="wireframe",
               line_width=5
    )
    # Add title to colorbar
    p.add_text(r"$\Delta \phi_M \rm (mV)$",
               position=position_text,
               font_size=14,
               color="black",
               viewport=True,
               orientation=-90,
    )

    # Set the camera position and save
    p.camera_position = camera_position
    p.screenshot(f"{fname}.png", transparent_background=True)
    p.close()

    # Plot glial potential in roi
    grid_glial_roi = grid_glial.clip_box(bounds=roi_bounds, invert=False)
    p = pyvista.Plotter(off_screen=True)

    # Add membrane potential
    p.add_mesh(grid_glial_roi,
               scalars="diff",
               scalar_bar_args=sargs,
               cmap=cmap_glial,
               clim=clim,
               show_scalar_bar=False,
    )
    # Add box ROI
    p.add_mesh(roi_box,
              color="black",
              style="wireframe",
              line_width=5,
              show_edges=True
    )
    # Add membrane point in ROI
    p.add_mesh(roi_point,
               color=c_point,
               point_size=50,
               render_points_as_spheres=True
    )

    # Set the camera position and zoom in and save
    p.camera_position = camera_position
    p.camera.zoom(5)
    p.screenshot(f"{fname}_roi.png", transparent_background=True)
    p.close()

    return


def plot_ECS_concentration(fname, ion, ECS_bounds, roi_box, origin, grid_ECS, \
                           grid_ECS_init, custom_labels, cmap, clim):

    slice_ECS = grid_ECS.slice(normal='x', origin=origin)
    slice_roi_box = roi_box.slice(normal='x', origin=origin)
    clipped_ECS = slice_ECS.clip_box(bounds=ECS_bounds, invert=False)

    position_bar=[0.83, 0.25]
    position_text=(0.88, 0.60)
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
    p.add_mesh(clipped_ECS,
               cmap=cmap,
               scalar_bar_args=sargs,
               annotations=custom_labels,
               clim=clim
    )
    # Add ROI box
    p.add_mesh(slice_roi_box,
               color="black", 
               style="wireframe",
               line_width=5
    )
    # Add title to color bar
    p.add_text(r"$[$" + f"{ion}" + r"$]_{\rm e}$ (mM)",
               position=position_text,
               orientation=-90,
               font_size=14,
               viewport=True
    )

    # Make pretty and save
    p.reset_camera()
    p.camera.zoom(1.0) # Increase zoom to 'crop' out the edges
    p.camera_position = "yz"
    p.screenshot(f"{fname}.png", transparent_background=True)
    p.close()

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

    # Get ROI
    x_L = config["x_L"]; x_U = config["x_U"]
    y_L = config["y_L"]; y_U = config["y_U"]
    z_L = config["z_L"]; z_U = config["z_U"]

    # Define ROI bounds and box
    roi_bounds = [x_L, x_U, y_L, y_U, z_L, z_U]
    roi_box = pyvista.Box(bounds=(x_L, x_U, y_L, y_U, z_L, z_U))

    # Get membrane point for plotting
    x_M = config["x_M"]
    y_M = config["y_M"]
    z_M = config["z_M"]
    # define point at glial membrane in roi
    roi_point_membrane = pyvista.PolyData([x_M, y_M, z_M])

    x_E = config["x_e"]
    y_E = config["y_e"]
    z_E = config["z_e"]
    # define point in ECS in roi
    roi_point_ECS = pyvista.PolyData([x_E, y_E, z_E])

    # Get center point (c,c,c)
    c = config["c"]

    # Define bounds for ECS plot
    x_L_E = x_L - 1100e-7; x_U_E = x_U + 1100e-7
    y_L_E = y_L - 1100e-7; y_U_E = y_U + 1100e-7
    z_L_E = z_L - 1100e-7; z_U_E = z_U + 1100e-7
    ECS_bounds = [x_L_E, x_U_E, y_L_E, y_U_E, z_L_E, z_U_E]

    # get filename and mesh name
    filename = f"../../{config['mesh_file_original']}"
    mesh_name = config['mesh_name']

    #times = [r't = 92.1 ms', r't = 92.6 ms', r't = 93.1 ms']

    dir = f"baseline_{mesh_name}"

    # Create directory for plots if it doesn't exist
    output_dir = Path(f"results/{dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set color maps for glial potential and ECS K+ concentration
    cmap_glial = seaborn.color_palette("inferno", as_cmap=True)
    cmap_ECS_K = seaborn.color_palette("crest", as_cmap=True)

    if mesh_name == "D2":
        # Set camera position for plotting mesh of domain D2
        camera_position = [
        (0.0011148767713648874, -0.0007533038582797973, -0.0000159169912591045),  # Position[cite: 3]
        (0.0002497464765838219, 0.0002511593568215165, 0.00025144041546809587),  # Focal Point[cite: 3]
        (-0.13157707833462606, 0.14758908754156708, -0.9802575853802773)          # View Up[cite: 3]
        ]

        clim_glial = [4.5, 5.6]
        custom_labels_glial = {5:"5"}

        # Plot ECS K field
        clim_ECS_K = [4, 11]
        custom_labels_ECS_K = {5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "11"}

    i = 1
    #for time_index in [184, 185]:
    for time_index in [24, 25]:

        # Get solution glial membrane potential at time time_index and time 0
        grid_glial = get_grid_field(dir, "results_mem_2", "phi_M_2", time_index)
        grid_glial_init = get_grid_field(dir, "results_mem_2", "phi_M_2", 0)
        # Remove small islands in plot
        ri_grid_glial = grid_glial.connectivity(extraction_mode='largest')
        ri_grid_glial_init = grid_glial_init.connectivity(extraction_mode='largest')

        # Plot glial membrane potential
        fname_glial = f"results/{dir}/glial_{i}"
        plot_glial_potential(fname_glial, roi_box, roi_bounds, \
                roi_point_membrane, ri_grid_glial, ri_grid_glial_init, \
                clim_glial, custom_labels_glial, camera_position)

        # Get solution ECS K+ concentration at time time_index and time 0
        grid_ECS = get_grid_field(dir, "results_sub_0", "c_K_0", time_index)
        grid_ECS_init = get_grid_field(dir, "results_sub_0", "c_K_0", 0)

        fname_ECS = f"results/{dir}/ECS_{i}"
        plot_ECS_concentration(fname_ECS, 'K', ECS_bounds, roi_box, \
                [x_M, c, c], grid_ECS, grid_ECS_init, custom_labels_ECS_K, 
                cmap_ECS_K, clim_ECS_K)

        i += 1

"""
# Calculate averages
for time_index in [index_1]:
    # Calculate average

    # ECS K+
    ARRAY_NAME = "c_K_0"
    grid_ECS = get_grid_field(dir, "results_sub_0", "c_K_0", time_index)
    bounds = [x_L, x_U, y_L, y_U, z_L, z_U]
    grid_ECS_roi = grid_ECS.clip_box(bounds, invert=False)
    avg_global_K_E = get_vw_average(grid_ECS, ARRAY_NAME)
    avg_roi_K_E = get_vw_average(grid_ECS_roi, ARRAY_NAME)

    # Plot the original (ghosted) and the slice
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(grid_ECS_roi)
    p.screenshot(f"ECS_roi.png", transparent_background=True)
    p.close()

    # Mem pot glial
    ARRAY_NAME = "phi_M_2"
    grid_glial = get_grid_field(dir, "results_mem_2", "phi_M_2", time_index)
    bounds = [x_L, x_U, y_L, y_U, z_L, z_U]
    grid_glial_roi = grid_glial.clip_box(bounds, invert=False)
    avg_global_phi_M = get_vw_average(grid_glial, ARRAY_NAME)
    avg_roi_phi_M = get_vw_average(grid_glial_roi, ARRAY_NAME)

    # Plot the original (ghosted) and the slice
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(grid_glial_roi)
    p.screenshot(f"glial_roi.png", transparent_background=True)
    p.close()
"""


