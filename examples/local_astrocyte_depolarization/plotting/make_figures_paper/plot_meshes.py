from pathlib import Path
import meshio
import pyvista

# Allow to plot empty meshes
pyvista.global_theme.allow_empty_mesh = True

COLORS = {
    "ECS": "#4e5f70",
    "neuron": "#16a085",
    "glial": "#ff67ff",
    "synapse_1": "#00ff00",
    "synapse_2": "#a0c991",
    "point": "#ffff00",
}

sargs = dict(
    title=r"$\rm [Na]_e$",
    n_labels=3,                # Number of labels
    fmt="%.2f",                # Decimal formatting
    font_family="arial",
    vertical=True,            # Horizontal orientation
    position_x=0.8,           # Move left/right (0 to 1)
    position_y=0.25,           # Move up/down (0 to 1)
    width=0.1,                 # Width of the bar
    height=0.6,                 # Height of the bar
    title_font_size=50,
    label_font_size=50,
)

# Region in which to apply the source term (cm)
x_L = 2000.0; x_U = 3000.0
y_L = 2000.0; y_U = 3000.0
z_L = 2200.0; z_U = 2600.0

x_M = 2683.0
y_M = 2889.0
z_M = 2206.0

# center point (c,c,c)
c = 2500

roi_bounds = [x_L, x_U, y_L, y_U, z_L, z_U]
roi_box = pyvista.Box(bounds=(x_L, x_U, y_L, y_U, z_L, z_U))

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
    p.add_mesh(slice_ECS, scalar_bar_args=sargs, color=COLORS['ECS'])
    p.add_mesh(slice_neuron, scalar_bar_args=sargs, color=COLORS['neuron'])
    p.add_mesh(slice_glial, scalar_bar_args=sargs, color=COLORS['glial'])
    p.add_mesh(slice_syn_1, scalar_bar_args=sargs, color=COLORS['synapse_1'])
    p.add_mesh(slice_syn_2, scalar_bar_args=sargs, color=COLORS['synapse_2'])
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

def plot_ECS(mesh_name, x, origin, grid_ECS):

    # Plot ECS
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(grid_ECS, scalar_bar_args=sargs, color=COLORS['ECS'])

    # Make pretty and save
    p.camera_position = 'yz'
    p.camera.azimuth += 225
    p.camera.elevation += 15
    p.reset_camera()
    p.screenshot(f"results/ECS_{mesh_name}.png", transparent_background=True)
    p.close()

def plot_astrocyte_synapse(mesh_name, x, origin, grid_glial, grid_syn_1, grid_syn_2):

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


def plot_neurons(mesh_name, x, origin, grid_neuron):

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

# Plot D1
filename = f"../../meshes/synapse_D1/meshes/mesh.xdmf"
mesh_name = 'D1'
grid_ECS = get_grid(filename, [1, 1])
grid_glial = get_grid (filename, [3, 3])        # glial cell of interest that has PAPs in ROI
grid_glial_other = get_grid (filename, [4, 4])  # other glial cell
grid_syn_1 = get_grid (filename, [5, 5])
grid_syn_2 = get_grid (filename, [39, 39])

# get grids for remaining neurons and add them together to one grid
grid_neuron = get_grid(filename, [2, 2]) \
            + get_grid(filename, [6, 38]) \
            + get_grid(filename, [40, 90])

plot_2D(mesh_name, 'x', [x_M, c, c], "yz", grid_ECS, grid_neuron, grid_glial + grid_glial_other, grid_syn_1, grid_syn_2)
plot_2D(mesh_name, 'y', [c, y_M, c], "xz", grid_ECS, grid_neuron, grid_glial + grid_glial_other, grid_syn_1, grid_syn_2)
plot_2D(mesh_name, 'z', [c, c, z_M], "xy", grid_ECS, grid_neuron, grid_glial + grid_glial_other, grid_syn_1, grid_syn_2)
plot_ECS(mesh_name, 'y', [c, x_M, c], grid_ECS)
plot_neurons(mesh_name, 'y', [c, x_M, c], grid_neuron)
plot_astrocyte_synapse(mesh_name, 'y', [c, x_M, c], grid_glial, grid_syn_1, grid_syn_2)

# Plot D2
filename = f"../../meshes/synapse_D2/meshes/mesh.xdmf"
mesh_name = 'D2'
grid_ECS = get_grid(filename, [1, 1])
grid_glial = get_grid (filename, [2, 2])        # glial cell of interest that has PAPs in ROI
grid_glial_other = get_grid (filename, [3, 3])  # other glial cell
grid_syn_1 = get_grid (filename, [4, 4])
grid_syn_2 = get_grid (filename, [47, 47])

# get grids for remaining neurons and add them together to one grid
grid_neuron = get_grid(filename, [5, 46]) + get_grid(filename, [48, 90])

plot_2D(mesh_name, 'x', [x_M, c, c], "yz", grid_ECS, grid_neuron, grid_glial + grid_glial_other, grid_syn_1, grid_syn_2)
plot_2D(mesh_name, 'y', [c, y_M, c], "xz", grid_ECS, grid_neuron, grid_glial + grid_glial_other, grid_syn_1, grid_syn_2)
plot_2D(mesh_name, 'z', [c, c, z_M], "xy", grid_ECS, grid_neuron, grid_glial + grid_glial_other, grid_syn_1, grid_syn_2)
plot_ECS(mesh_name, 'y', [c, x_M, c], grid_ECS)
plot_neurons(mesh_name, 'y', [c, x_M, c], grid_neuron)
plot_astrocyte_synapse(mesh_name, 'y', [c, x_M, c], grid_glial, grid_syn_1, grid_syn_2)
