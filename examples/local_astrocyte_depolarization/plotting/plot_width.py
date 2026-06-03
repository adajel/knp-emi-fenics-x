import pyvista
import yaml
import numpy as np
#from io_utils import read_xdmf_timeseries, xdmf_to_unstructuredGrid
import dolfinx
import adios4dolfinx.backends.xdmf.backend
from mpi4py import MPI

c_ECS = "#4e5f70"
c_neuron = "#16a085"
c_glial = "#ff67ff"
c_synapse_1 = "#00ff00"
c_synapse_2 = "#e1fae1"
c_point = "#ffff00"

#sargs = dict(
#    title=r"$\rm [Na]_e$",
#    n_labels=3,                # Number of labels
#    fmt="%.2f",                # Decimal formatting
#    font_family="arial",
#    vertical=True,            # Horizontal orientation
#    position_x=0.8,           # Move left/right (0 to 1)
#    position_y=0.25,           # Move up/down (0 to 1)
#    width=0.1,                 # Width of the bar
#    height=0.6,                 # Height of the bar
#    title_font_size=50,
#    label_font_size=50,
#)

# Region in which to apply the source term (cm)
#x_L = 2100.0; x_U = 2900.0
#y_L = 2100.0; y_U = 2900.0
#z_L = 2100.0; z_U = 2500.0

x_L = 2000; x_U = 3000
y_L = 2000; y_U = 3000
z_L = 2100; z_U = 2700

roi_bounds = [x_L, x_U, y_L, y_U, z_L, z_U]
roi_box = pyvista.Box(bounds=(x_L, x_U, y_L, y_U, z_L, z_U))

# Coordinates of point
x_M = 2683.0
y_M = 2889.0
z_M = 2206.0

# Center point (c,c,c) in mesh
c = 2500.0

def get_grid(finame, funame, time):
    # Read mesh from file
    filename = f"../results/make_mesh/{finame}.xdmf"
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
    float_stamps = np.array(timestamps, dtype=np.float64)
    pos = np.flatnonzero(np.isclose(float_stamps, time))
    assert len(pos) == 1

    # Read data from file
    p0 = adios4dolfinx.read_point_data(
        filename, f"{funame}", grid, timestamps[pos[0]], backend="xdmf")
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(p0.function_space))
    grid.point_data[f"{funame}"] = p0.x.array

    return grid

def plot_ECS_width(x, clim, origin, camera_position, grid_syn_1, grid_syn_2, grid_ECS_width):

    slice_plane_ECS_width = grid_ECS_width.slice(normal=x, origin=origin)
    slice_plane_syn_1 = grid_syn_1.slice(normal=x, origin=origin)
    slice_plane_syn_2 = grid_syn_2.slice(normal=x, origin=origin)
    slice_plane_glial = grid_glial.slice(normal=x, origin=origin)
    slice_plane_neuron = grid_neuron.slice(normal=x, origin=origin)
    slice_roi_box = roi_box.slice(normal=x, origin=origin)

    # Zoom in to ROI
    clipped_ECS_width = slice_plane_ECS_width.clip_box(bounds=roi_bounds, invert=False)
    clipped_syn_1 = slice_plane_syn_1.clip_box(bounds=roi_bounds, invert=False)
    clipped_syn_2 = slice_plane_syn_2.clip_box(bounds=roi_bounds, invert=False)
    clipped_glial = slice_plane_glial.clip_box(bounds=roi_bounds, invert=False)
    clipped_neuron = slice_plane_neuron.clip_box(bounds=roi_bounds, invert=False)

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

    p.add_mesh(slice_plane_ECS_width,
               cmap="inferno",
               clim=clim,
               scalar_bar_args=sargs,
               annotations=custom_labels)

    p.add_mesh(slice_plane_syn_1, color=c_synapse_1)
    p.add_mesh(slice_plane_glial, color=c_glial)
    p.add_mesh(slice_plane_neuron, color=c_neuron)
    if x == 'x' or x == 'y':
        p.add_mesh(slice_plane_syn_2, color=c_synapse_2)
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

    # 4. Save the screenshot
    p.screenshot(f"results/ECS_width_{x}.png", transparent_background=True)
    p.close()

    # Zoom in
    p = pyvista.Plotter(off_screen=True)

    p.add_mesh(clipped_ECS_width,
               cmap="inferno",
               clim=clim,
               scalar_bar_args=sargs,
               annotations=custom_labels,
               show_scalar_bar=False,
               )

    p.add_mesh(slice_roi_box, color="black", style="wireframe", line_width=5, label="ROI")
    p.add_mesh(clipped_glial, color=c_glial)
    p.add_mesh(clipped_neuron, color=c_neuron)
    p.add_mesh(clipped_syn_1, color=c_synapse_1)
    if x == 'x' or x == 'y':
        p.add_mesh(clipped_syn_2, color=c_synapse_2)

    # Focus the camera tightly on the object
    p.camera_position = camera_position

    # 4. Save the screenshot
    p.screenshot(f"results/ECS_width_roi_{x}.png", transparent_background=True)
    p.close()

def plot_glial_width(x, clim, origin, camera_position, grid_glial_width):

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

    p.add_mesh(grid_glial_width, cmap="inferno", clim=clim, scalar_bar_args=sargs, annotations=custom_labels)
    #p.add_mesh(grid_glial, color=c_glial)
    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5,
            label="ROI", show_edges=True)

    p.add_text(
        "Local width glial (nm)",
        position=position, # Adjust X and Y as needed
        orientation=-90,   # Rotate text 90 degrees
        font_size=14,
        viewport=True
    )

    # Fix camera position and zoom
    p.camera_position = 'yz'
    p.camera.azimuth += 225
    p.camera.elevation += 15
    p.reset_camera()

    # Save screenshot
    p.screenshot(f"results/glial_width.png", transparent_background=True)
    p.close()

    # Plot membrane potential in ROI
    p = pyvista.Plotter(off_screen=True)

    clipped_glial_width = grid_glial_width.clip_box(bounds=roi_bounds, invert=False)

    p.add_mesh(clipped_glial_width,
               cmap="inferno",
               clim=clim,
               scalar_bar_args=sargs,
               annotations=custom_labels, 
               show_scalar_bar=False,
               )

    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5,
            label="ROI", show_edges=True)

    # Fix camera position and zoom
    p.camera_position = 'yz'
    p.camera.azimuth += 225
    p.camera.elevation += 15
    p.reset_camera()

    # Save screenshot
    p.screenshot(f"results/roi_glial_width.png", transparent_background=True)
    p.close()

def plot_roi_with_point(x, origin, camera_position, grid_glial):

    roi_point = pyvista.PolyData([x_M, y_M, z_M])
    clipped_glial = grid_glial.clip_box(bounds=roi_bounds, invert=False)

    # Make plot zoom in glial
    p = pyvista.Plotter(off_screen=True)

    p.add_mesh(clipped_glial, color=c_glial)

    p.add_mesh(roi_point,
               color=c_point,
               point_size=30, 
               render_points_as_spheres=True)

    p.add_mesh(roi_box,
               color="black",
               style="wireframe",
               line_width=5,
               label="ROI",
               show_edges=True)

    # Fix camera position and zoom
    p.camera_position = 'yz'
    p.camera.azimuth += 140 + 90 + 90
    p.camera.elevation += 20
    p.reset_camera()

    # Save screenshot
    p.screenshot(f"results/roi_with_point.png", transparent_background=True)
    p.close()

# Get grids
time = 0.1
grid_ECS = get_grid("results_sub_0", "c_K_0", time)
grid_neuron = get_grid("results_sub_1", "c_K_1", time)
grid_glial = get_grid("results_sub_2", "c_K_2", time)
grid_syn_1 = get_grid("results_sub_3", "c_K_3", time)
grid_syn_2 = get_grid("results_sub_4", "c_K_4", time)

# Read and plot ECS width
grid_ECS_width = pyvista.read('ecs.vtk')
clim=[10, 260]
plot_ECS_width('x', clim, [x_M, c, c], "yz", grid_syn_1, grid_syn_2, grid_ECS_width)

# Read and plot glial width
grid_glial_width = pyvista.read('glial.vtk')
clim=[40, 360]
plot_glial_width('z', clim, [c, c, z_M], "xy", grid_glial_width)

# Make plot of glial cell in ROI with point
plot_roi_with_point('x', [x_M, c, c], "xy", grid_glial)
