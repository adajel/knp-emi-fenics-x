from pathlib import Path
import meshio
import pyvista
import argparse
import yaml

# Allow to plot empty meshes
pyvista.global_theme.allow_empty_mesh = True

COLORS = {
    "ECS": "#4e5f70",
    "neuron": "#16a085",
    "glial": "#ff67ff",
    "synapse_1": "#00ff00",
    "synapse_2": "#a0c991",
    "point": "#FF073A",
}

def get_grid(filename, mesh_tags):

    # read file and convert meshio Mesh to PyVista UnstructuredGrid
    msh = meshio.read(filename)
    mesh = pyvista.from_meshio(msh)

    # Extract separate regions
    subdomain_grid = mesh.threshold(mesh_tags, scalars='marker')

    return subdomain_grid

def plot_2D(mesh_name, x, origin, camera_position, grid_ECS, grid_neuron, grid_glial, grid_syn_1, grid_syn_2):

    # slice grids
    slice_ECS = grid_ECS.slice(normal=x, origin=origin)
    slice_neuron = grid_neuron.slice(normal=x, origin=origin)
    slice_glial = grid_glial.slice(normal=x, origin=origin)
    slice_syn_1 = grid_syn_1.slice(normal=x, origin=origin)
    slice_syn_2 = grid_syn_2.slice(normal=x, origin=origin)
    slice_roi_box = roi_box.slice(normal=x, origin=origin)

    # clip grids (zoom in on ROI)
    clipped_ECS = slice_ECS.clip_box(bounds=roi_bounds, invert=False)
    clipped_glial = slice_glial.clip_box(bounds=roi_bounds, invert=False)
    clipped_neuron = slice_neuron.clip_box(bounds=roi_bounds, invert=False)
    clipped_syn_1 = slice_syn_1.clip_box(bounds=roi_bounds, invert=False)
    clipped_syn_2 = slice_syn_2.clip_box(bounds=roi_bounds, invert=False)

    # Plot 2D slices
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(slice_ECS, color=COLORS['ECS'])
    p.add_mesh(slice_neuron, color=COLORS['neuron'])
    p.add_mesh(slice_glial, color=COLORS['glial'])
    p.add_mesh(slice_syn_1, color=COLORS['synapse_1'])
    p.add_mesh(slice_syn_2, color=COLORS['synapse_2'])
    p.add_mesh(slice_roi_box, color="black", style="wireframe", line_width=3)

    # Make pretty and save
    p.reset_camera()
    p.camera.zoom(1.0)
    p.camera_position = camera_position
    p.screenshot(f"results/2D_{x}_{mesh_name}.png", transparent_background=True)
    p.close()

    # Plot 2D slices zoom in on ROI
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(clipped_ECS, color=COLORS['ECS'])
    p.add_mesh(clipped_glial, color=COLORS['glial'])
    p.add_mesh(clipped_syn_1, color=COLORS['synapse_1'])
    if x == 'z':
        p.add_mesh(clipped_syn_2, color=COLORS['synapse_2'])
        p.add_mesh(clipped_neuron, color=COLORS['neuron'])
    if x == 'y':
        p.add_mesh(clipped_neuron, color=COLORS['neuron'])

    # Make pretty and save
    p.camera_position = camera_position
    p.reset_camera()
    p.camera.zoom(2.0) # Increase zoom to 'crop' out the edges
    p.screenshot(f"results/2D_roi_{x}_{mesh_name}.png", transparent_background=True)
    p.close()

    return


def plot_neurons(mesh_name, grid_neuron):

    # Plot neurons
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(grid_neuron, color=COLORS['neuron'])
    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5)

    # Make pretty and save
    p.camera_position = 'yz'
    p.camera.azimuth += 225
    p.camera.elevation += 15
    p.reset_camera()
    p.screenshot(f"results/neurons_{mesh_name}.png", transparent_background=True)
    p.close()

    return


def plot_ECS(mesh_name, grid_ECS):

    # Plot ECS
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(grid_ECS, color=COLORS['ECS'])

    # Make pretty and save
    p.camera_position = 'yz'
    p.camera.azimuth += 225
    p.camera.elevation += 15
    p.reset_camera()
    p.screenshot(f"results/ECS_{mesh_name}.png", transparent_background=True)
    p.close()

def plot_astrocyte_synapse(mesh_name, grid_glial, grid_syn_1, grid_syn_2):

    # Clip grids to zoom in on ROI
    clipped_glial = grid_glial.clip_box(bounds=roi_bounds, invert=False)
    clipped_syn_1 = grid_syn_1.clip_box(bounds=roi_bounds, invert=False)
    clipped_syn_2 = grid_syn_2.clip_box(bounds=roi_bounds, invert=False)

    # Plot astrocyte and synapse
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(grid_glial, color=COLORS['glial'])
    p.add_mesh(grid_syn_1, color=COLORS['synapse_1'])
    p.add_mesh(grid_syn_2, color=COLORS['synapse_2'])
    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5)

    # Make pretty and save
    p.camera_position = 'yz'
    if mesh_name == "D1":
        p.camera.azimuth += 225
    elif mesh_name == "D2":
        p.camera.azimuth += 225-180-90
    p.camera.elevation += 15
    p.reset_camera()
    p.screenshot(f"results/astrocyte_synapse_{mesh_name}.png", transparent_background=True)
    p.close()

    # Plot astrocyte and synapse zoom in on ROI
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(clipped_glial, color=COLORS['glial'])
    p.add_mesh(clipped_syn_1, color=COLORS['synapse_1'])
    p.add_mesh(clipped_syn_2, color=COLORS['synapse_2'])
    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5)

    # Make pretty and save
    p.camera_position = 'yz'
    p.camera.azimuth += 225-180-90
    p.camera.elevation += 15
    p.reset_camera()
    p.screenshot(f"results/astrocyte_synapse_roi_{mesh_name}.png", transparent_background=True)
    p.close()

    return

def visualize_plotting_points(mesh_name, config, grid_glial, grid_syn_1, grid_syn_2):

    # Get membrane point for plotting
    x_M = config["x_M"]*1.0e7
    y_M = config["y_M"]*1.0e7
    z_M = config["z_M"]*1.0e7
    # Get ICS point for plotting
    x_i = config["x_i"]*1.0e7
    y_i = config["y_i"]*1.0e7
    z_i = config["z_i"]*1.0e7
    # Get ECS point for plotting
    x_e = config["x_e"]*1.0e7
    y_e = config["y_e"]*1.0e7
    z_e = config["z_e"]*1.0e7

    roi_point_M = pyvista.PolyData([x_M, y_M, z_M])
    roi_point_i = pyvista.PolyData([x_i, y_i, z_i])
    roi_point_e = pyvista.PolyData([x_e, y_e, z_e])

    # Clip grids to zoom in on ROI
    clipped_glial = grid_glial.clip_box(bounds=roi_bounds, invert=False)
    clipped_syn_1 = grid_syn_1.clip_box(bounds=roi_bounds, invert=False)
    clipped_syn_2 = grid_syn_2.clip_box(bounds=roi_bounds, invert=False)

    # Plot astrocyte and synapse zoom in on ROI
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(clipped_glial, color=COLORS['glial'], opacity=0.7)
    p.add_mesh(clipped_syn_1, color=COLORS['synapse_1'], opacity=0.4)
    p.add_mesh(clipped_syn_2, color=COLORS['synapse_2'], opacity=0.4)
    #p.add_mesh(roi_box, color="black", style="wireframe", line_width=5)

    p.add_mesh(roi_point_M, color=COLORS['point'], point_size=20, render_points_as_spheres=True)
    p.add_mesh(roi_point_i, color=COLORS['point'], point_size=20, render_points_as_spheres=True)
    p.add_mesh(roi_point_e, color=COLORS['point'], point_size=20, render_points_as_spheres=True)

    # Make pretty and save
    p.camera_position = 'yz'
    p.camera.azimuth += 225-180-90
    p.camera.elevation += 15
    p.reset_camera()
    p.screenshot(f"results/point_roi_{mesh_name}.png", transparent_background=True)
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
    x_L = config["x_L"]*1.0e7; x_U = config["x_U"]*1.0e7
    y_L = config["y_L"]*1.0e7; y_U = config["y_U"]*1.0e7
    z_L = config["z_L"]*1.0e7; z_U = config["z_U"]*1.0e7

    # Define ROI bounds and box
    roi_bounds = [x_L, x_U, y_L, y_U, z_L, z_U]
    roi_box = pyvista.Box(bounds=(x_L, x_U, y_L, y_U, z_L, z_U))

    # Get membrane point for plotting
    x_M = config["x_M"]*1.0e7
    y_M = config["y_M"]*1.0e7
    z_M = config["z_M"]*1.0e7
    # Get center point (c,c,c)
    c = config["c"]*1.0e7

    # get filename and mesh name
    filename = f"../../{config['mesh_file_original']}"
    mesh_name = config['mesh_name']

    # get ECS grid
    grid_ECS = get_grid(filename, [1, 1])

    # get neuronal and glial grids
    if mesh_name == 'D1':
        grid_glial_roi = get_grid(filename, [3, 3])        # glial cell of interest that has PAPs in ROI
        grid_glial_other = get_grid(filename, [4, 4])  # other glial cell
        grid_syn_1 = get_grid(filename, [5, 5])
        grid_syn_2 = get_grid(filename, [39, 39])
        # get grids for remaining neurons and add them together to one grid
        grid_neuron_all = get_grid(filename, [2, 2]) + get_grid(filename, [6, 38]) + get_grid(filename, [40, 90])
        grid_glial_all = grid_glial_roi + grid_glial_other

    elif mesh_name == 'D2':
        grid_glial_roi = get_grid(filename, [2, 2])        # glial cell of interest that has PAPs in ROI
        grid_glial_other = get_grid(filename, [3, 3])  # other glial cell
        grid_syn_1 = get_grid(filename, [4, 4])
        grid_syn_2 = get_grid(filename, [47, 47])
        # get grids for remaining neurons and add them together to one grid
        grid_neuron_all = get_grid(filename, [5, 46]) + get_grid(filename, [48, 90])
        grid_glial_all = grid_glial_roi + grid_glial_other

    elif mesh_name == 'D3':
        grid_glial_roi = get_grid(filename, [2, 2])        # glial cell of interest that has PAPs in ROI (only one glial cell in this geometry)
        grid_syn_1 = get_grid(filename, [26, 26])
        grid_syn_2 = get_grid(filename, [30, 30])
        # get grids for remaining neurons and add them together to one grid
        grid_neuron_all = get_grid(filename, [3, 25]) + get_grid(filename, [27, 29]) + get_grid(filename, [31, 90])
        grid_glial_all = grid_glial_roi

    plot_2D(mesh_name, 'x', [x_M, c, c], "yz", grid_ECS, grid_neuron_all, grid_glial_all, grid_syn_1, grid_syn_2)
    plot_2D(mesh_name, 'y', [c, y_M, c], "xz", grid_ECS, grid_neuron_all, grid_glial_all, grid_syn_1, grid_syn_2)
    plot_2D(mesh_name, 'z', [c, c, z_M], "xy", grid_ECS, grid_neuron_all, grid_glial_all, grid_syn_1, grid_syn_2)

    plot_ECS(mesh_name, grid_ECS)
    plot_neurons(mesh_name, grid_neuron_all)
    plot_astrocyte_synapse(mesh_name, grid_glial_roi, grid_syn_1, grid_syn_2)

    visualize_plotting_points(mesh_name, config, grid_glial_roi, grid_syn_1, grid_syn_2)
