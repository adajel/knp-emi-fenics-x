import pyvista
import numpy as np
import meshio
import argparse
import yaml

c_ECS = "#4e5f70"
c_neuron = "#16a085"
c_glial = "#ff67ff"
c_synapse_1 = "#00ff00"
c_synapse_2 = "#e1fae1"
c_point = "#ffff00"

# Allow plotting empty meshes
pyvista.global_theme.allow_empty_mesh = True

def get_grid(filename, mesh_tags):

    # read file and convert meshio Mesh to PyVista UnstructuredGrid
    msh = meshio.read(filename)
    mesh = pyvista.from_meshio(msh)

    # Extract separate regions
    subdomain_grid = mesh.threshold(mesh_tags, scalars='marker')

    return subdomain_grid

def print_avg(mesh, box_bounds, label):
    """ Print avg of local width """

    scalar_name = 'local_width'

    working_mesh_roi = mesh.clip_box(bounds=box_bounds, invert=False)
    working_mesh_global = mesh.copy()

    true_spatial_average = []

    for working_mesh in [working_mesh_global, working_mesh_roi]:

        # Ensure data is mapped onto cells (elements) for volume weighting
        if scalar_name in working_mesh.point_data:
            working_mesh = working_mesh.point_data_to_cell_data()

        # Compute cell sizes explicitly
        mesh_with_sizes = working_mesh.compute_cell_sizes()

        # Extract arrays and enforce positive volumes using np.abs()
        volumes = np.abs(mesh_with_sizes.cell_data['Volume'])
        scalars = mesh_with_sizes.cell_data[scalar_name]

        # Final math
        true_total_volume = np.sum(volumes)
        true_spatial_average.append(np.sum(scalars * volumes) / true_total_volume)

    formatted_label = f"{label:<18}"
    print(f"{formatted_label} {true_spatial_average[0]:.0f} ({true_spatial_average[1]:.0f}) nm")

    return

def plot_local_width_ECS(mesh_name, x, clim, origin, camera_position, grid_syn_1, grid_syn_2, grid_ECS_width):

    slice_ECS_width = grid_ECS_width.slice(normal=x, origin=origin)
    slice_syn_1 = grid_syn_1.slice(normal=x, origin=origin)
    slice_syn_2 = grid_syn_2.slice(normal=x, origin=origin)
    slice_glial = grid_glial.slice(normal=x, origin=origin)
    slice_neuron = grid_neuron.slice(normal=x, origin=origin)
    slice_roi_box = roi_box.slice(normal=x, origin=origin)

    # Zoom in to ROI
    clipped_ECS_width = slice_ECS_width.clip_box(bounds=roi_bounds, invert=False)
    clipped_syn_1 = slice_syn_1.clip_box(bounds=roi_bounds, invert=False)
    clipped_syn_2 = slice_syn_2.clip_box(bounds=roi_bounds, invert=False)
    clipped_glial = slice_glial.clip_box(bounds=roi_bounds, invert=False)
    clipped_neuron = slice_neuron.clip_box(bounds=roi_bounds, invert=False)

    custom_labels = {
        10: "10",
        50: "50",
        100: "100",
        150: "150",
        200: "200",
        250: "250",
    }

    if x == 'x':
        position_x=0.82
        position_y=0.175
        position=(0.87, 0.54)
    elif x == 'y':
        position_x=0.77
        position_y=0.33
        position=(0.82, 0.63)
    if x == 'z':
        position_x=0.75
        position_y=0.35
        position=(0.80, 0.65)

    sargs = dict(
        title="",
        vertical=True,
        position_x=position_x, 
        position_y=position_y,
        height=0.35,
        n_labels=0,
        width=0.1,
        label_font_size=25,
        shadow=True,
        fmt="%.0f", # Decimal formatting
    )

    # Plot ECS width
    p = pyvista.Plotter(off_screen=True)

    p.add_mesh(slice_ECS_width,
               scalars="local_width",
               cmap="inferno",
               clim=clim,
               scalar_bar_args=sargs,
               annotations=custom_labels)

    p.add_mesh(slice_syn_1, color=c_synapse_1)
    p.add_mesh(slice_glial, color=c_glial)
    p.add_mesh(slice_neuron, color=c_neuron)
    if x == 'x' or x == 'y':
        p.add_mesh(slice_syn_2, color=c_synapse_2)
    p.add_mesh(slice_roi_box, color="black", style="wireframe", line_width=5, label="ROI")

    p.add_text(
        "Local ECS width (nm)",
        position=position, # Adjust X and Y as needed
        orientation=-90,       # Rotate text 90 degrees
        font_size=14,
        viewport=True
    )

    # Focus the camera tightly on the object
    p.camera_position = camera_position

    # Make pretty and save
    p.screenshot(f"results/local_width_ECS_{mesh_name}.png", transparent_background=True)
    p.close()
    p = pyvista.Plotter(off_screen=True)

    p.add_mesh(clipped_ECS_width,
               scalars="local_width",
               cmap="inferno",
               clim=clim,
               scalar_bar_args=sargs,
               annotations=custom_labels,
               show_scalar_bar=False,
               )

    p.add_mesh(slice_roi_box, color="black", style="wireframe", line_width=5, label="ROI")
    p.add_mesh(clipped_glial, color=c_glial)
    #p.add_mesh(clipped_neuron, color=c_neuron)
    p.add_mesh(clipped_syn_1, color=c_synapse_1)
    if x == 'x' or x == 'y':
        p.add_mesh(clipped_syn_2, color=c_synapse_2)

    # Focus the camera tightly on the object
    p.camera_position = camera_position

    # 4. Save the screenshot
    p.screenshot(f"results/local_width_ECS_roi_{mesh_name}.png", transparent_background=True)
    p.close()

def plot_local_width_glial(mesh_name, x, clim, origin, camera_position, grid_glial_width):

    # 
    clipped_glial_width = grid_glial_width.clip_box(bounds=roi_bounds, invert=False)

    custom_labels = {
        50: "50",
        150: "150",
        250: "250",
        350: "350",
    }

    position_x=0.85
    position_y=0.30
    position=(0.9, 0.66)

    sargs = dict(
        title="",
        vertical=True,
        position_x=position_x,
        position_y=position_y,
        height=0.35,
        n_labels=0,
        width=0.1,
        label_font_size=25,
        shadow=True,
        fmt="%.0f", # Decimal formatting
    )

    # Plot global membrane potential
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(grid_glial_width,
               scalars="local_width",
               cmap="inferno",
               clim=clim,
               scalar_bar_args=sargs,
               annotations=custom_labels)
    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5,
            label="ROI", show_edges=True)

    p.add_text(
        "Local width glial (nm)",
        position=position, # Adjust X and Y as needed
        orientation=-90,   # Rotate text 90 degrees
        font_size=14,
        viewport=True
    )

    # Make pretty and save
    p.camera_position = 'yz'
    p.camera.azimuth += 225
    p.camera.elevation += 15
    p.reset_camera()
    p.screenshot(f"results/local_width_glial_{mesh_name}.png", transparent_background=True)
    p.close()

    # Plot membrane potential in ROI
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(clipped_glial_width,
               scalars="local_width",
               cmap="inferno",
               clim=clim,
               scalar_bar_args=sargs,
               annotations=custom_labels, 
               show_scalar_bar=False,
               )
    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5,
            label="ROI", show_edges=True)

    # Make pretty and save
    p.camera_position = 'yz'
    p.camera.azimuth += 225
    p.camera.elevation += 15
    p.reset_camera()
    p.screenshot(f"results/local_width_glial_roi_{mesh_name}.png", transparent_background=True)
    p.close()

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
    roi_box = pyvista.Box(bounds=(x_L, x_U, y_L, y_U, z_L, z_U))
    roi_bounds = [x_L, x_U, y_L, y_U, z_L, z_U]

    # Get membrane point for plotting
    x_M = config["x_M"]*1.0e7
    y_M = config["y_M"]*1.0e7
    z_M = config["z_M"]*1.0e7
    # Get center point (c,c,c)
    c = config["c"]*1.0e7

    # get filename and mesh name
    filename = f"../../{config['mesh_file_original']}"
    mesh_name = config['mesh_name']

    # Get cell tags
    tag_glial = config['tag_glial']
    tag_glial_other = config['tag_glial_other']
    tag_syn_pre = config['tag_syn_pre']
    tag_syn_post = config['tag_syn_post']

    grid_ECS = get_grid(filename, [1, 1])
    grid_glial = get_grid (filename, [tag_glial, tag_glial]) + get_grid (filename, [tag_glial_other, tag_glial_other])
    grid_syn_1 = get_grid (filename, [tag_syn_pre, tag_syn_pre])
    grid_syn_2 = get_grid (filename, [tag_syn_post, tag_syn_post])

    # Find all other tags (e.g. tags for remaining neurons)
    full_range = set(range(2, 91))
    tags_neurons = sorted(full_range - set([tag_glial, tag_glial_other, tag_syn_pre, tag_syn_post]))

    # get grids for remaining neurons and add them together to one grid
    grid_neuron = get_grid(filename, [tags_neurons[0], tags_neurons[0]])
    for tag in tags_neurons[1:]:
        grid_neuron += get_grid(filename, [tag, tag])

    # Read and plot local width ECS
    grid_ECS_width = pyvista.read(f'results/ecs_{mesh_name}.vtk')
    clim=[10, 250]
    plot_local_width_ECS(mesh_name, 'x', clim, [x_M, c, c], "yz", grid_syn_1, grid_syn_2, grid_ECS_width)

    # Read and plot local width glial
    grid_glial_width = pyvista.read(f'results/glial_{mesh_name}.vtk')
    clim=[20, 370]
    plot_local_width_glial(mesh_name, 'z', clim, [c, c, z_M], "xy", grid_glial_width)

    print(f"Local width averages for mesh {mesh_name}:")
    print("---------------------------------")
    print_avg(grid_glial_width, roi_bounds, "Avg. glial:")
    print_avg(grid_ECS_width, roi_bounds, "Avg. ECS:")
