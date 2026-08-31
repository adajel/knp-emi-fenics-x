import pyvista
import numpy as np
import meshio

c_ECS = "#4e5f70"
c_neuron = "#16a085"
c_glial = "#ff67ff"
c_synapse_1 = "#00ff00"
c_synapse_2 = "#e1fae1"
c_point = "#ffff00"

x_L = 2000; x_U = 3000
y_L = 2000; y_U = 3000
z_L = 2200; z_U = 2600

roi_bounds = [x_L, x_U, y_L, y_U, z_L, z_U]
roi_box = pyvista.Box(bounds=(x_L, x_U, y_L, y_U, z_L, z_U))

# Coordinates of point
x_M = 2608
y_M = 2859
z_M = 2184

# Center point in domain
c = 2500.0

def get_grid(filename, mesh_tags):

    # read file and convert meshio Mesh to PyVista UnstructuredGrid
    msh = meshio.read(filename)
    mesh = pyvista.from_meshio(msh)

    # Extract separate regions
    subdomain_grid = mesh.threshold(mesh_tags, scalars='marker')

    return subdomain_grid

def print_vw_avg(mesh, box_bounds=None):

    scalar_name = 'local_width'

    # Apply box clip if bounds are provided
    if box_bounds is not None:
        working_mesh = mesh.clip_box(bounds=box_bounds, invert=False)
    else:
        working_mesh = mesh.copy()

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
    true_spatial_average = np.sum(scalars * volumes) / true_total_volume

    print("------")
    print(f"Calculated Volume : {true_total_volume:.4f}")
    print(f"Spatial Average   : {true_spatial_average:.4f}")

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

# Plot D1
filename = f"../../meshes/synapse_D1/meshes/mesh.xdmf"
mesh_name = 'D1'
grid_ECS = get_grid(filename, [1, 1])
grid_glial = get_grid (filename, [3, 3]) + get_grid (filename, [4, 4])
grid_syn_1 = get_grid (filename, [5, 5])
grid_syn_2 = get_grid (filename, [39, 39])

# get grids for remaining neurons and add them together to one grid
grid_neuron = get_grid(filename, [2, 2]) \
            + get_grid(filename, [6, 38]) \
            + get_grid(filename, [40, 90])

# Read and plot local width ECS
grid_ECS_width = pyvista.read('results/ecs_D1.vtk')
clim=[10, 250]
plot_local_width_ECS(mesh_name, 'x', clim, [x_M, c, c], "yz", grid_syn_1, grid_syn_2, grid_ECS_width)

# Read and plot local width glial
grid_glial_width = pyvista.read('results/glial_D1.vtk')
clim=[20, 370]
plot_local_width_glial(mesh_name, 'z', clim, [c, c, z_M], "xy", grid_glial_width)

print_vw_avg(grid_glial_width)
print_vw_avg(grid_glial_width, roi_bounds)
print_vw_avg(grid_ECS_width)
print_vw_avg(grid_ECS_width, roi_bounds)
