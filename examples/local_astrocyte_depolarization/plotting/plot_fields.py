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

c_ECS = "#4e5f70"
c_neuron = "#16a085"
c_glial = "#ff67ff"
c_point = "#00ff00"

sargs_ECS = dict(
    title=" ",
    n_labels=5,                # Number of labels
    fmt="%.2f",                # Decimal formatting
    font_family="arial",
    vertical=True,             # Horizontal orientation
    position_x=0.85,           # Move left/right (0 to 1)
    position_y=0.30,           # Move up/down (0 to 1)
    width=0.08,                # Width of the bar
    height=0.4,                # Height of the bar
    title_font_size=20,
    label_font_size=20,
    color='black',
)

sargs_glial = dict(
    title=" ",
    n_labels=5,                # Number of labels
    fmt="%.2f",                # Decimal formatting
    font_family="arial",
    vertical=True,             # Horizontal orientation
    position_x=0.85,           # Move left/right (0 to 1)
    position_y=0.30,           # Move up/down (0 to 1)
    width=0.08,                # Width of the bar
    height=0.4,                # Height of the bar
    title_font_size=20,
    label_font_size=20,
    color='black',
)

cmap_ECS = "cool"
cmap_glial = "plasma"
#cmap_glial = "cool"

# Region in which to apply the source term (cm)
x_L = 2100e-7; x_U = 2900e-7
y_L = 2100e-7; y_U = 2900e-7
z_L = 2100e-7; z_U = 2500e-7

x_M = 2683e-7
y_M = 2889e-7
z_M = 2206e-7

# center point (c,c,c)
c = 2500e-7

def get_grid_mesh(finame, funame):
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

    # Get time based on provided index
    time = float(timestamps[0])

    float_stamps = np.array(timestamps, dtype=np.float64)
    pos = np.flatnonzero(np.isclose(float_stamps, time))
    assert len(pos) == 1

    # Read data from file
    p0 = adios4dolfinx.read_point_data(
        filename, f"{funame}", grid, timestamps[pos[0]], backend="xdmf")
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(p0.function_space))
    grid.point_data[f"{funame}"] = p0.x.array

    return grid

def get_grid_field(finame, funame, time_index):
    # Read mesh from file
    filename = f"../results/baseline_double_mem_cur/{finame}.xdmf"
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

    float_stamps = np.array(timestamps, dtype=np.float64)
    pos = np.flatnonzero(np.isclose(float_stamps, time))
    assert len(pos) == 1

    # Read data from file
    p0 = adios4dolfinx.read_point_data(
        filename, f"{funame}", grid, timestamps[pos[0]], backend="xdmf")
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(p0.function_space))
    grid.point_data[f"{funame}"] = p0.x.array

    return grid

def plot_ECS_K(grid_ECS, i):

    # Define box marking region of interest
    roi_box = pyvista.Box(bounds=(x_L, x_U, y_L, y_U, z_L, z_U))

    slice_plane_ECS = grid_ECS.slice(normal='x')
    slice_plane_roi = roi_box.slice(normal='x')

    # Make full 3D plot
    p = pyvista.Plotter(off_screen=True)

    p.add_mesh(slice_plane_ECS, scalar_bar_args=sargs_ECS, cmap=cmap_ECS)
    p.add_mesh(slice_plane_roi, color="black", style="wireframe", line_width=3, show_edges=True)

    # Fix camera position and zoom
    p.camera_position = 'yz'

    # add title to colorbar
    p.add_text(
        r"$[K]_e (mM)$",
        position=(0.83, 0.45),      # Right side, halfway up
        orientation=-270,          # Rotate 90 degrees clockwise
        font_size=11,
        color="black",
        viewport=True              # Uses the 0-1 coordinate system
    )

    # Save screenshot
    p.screenshot(f"results/ECS_K.png", transparent_background=True)
    p.close()

def plot_astrocyte_potential_ECS_embedding(grid_ECS, grid_neuron, grid_glial, i):

    box_ECS = grid_ECS.clip_box(bounds=[0, 3000e-7, 0, 3000e-7, 0, 5000e-7], invert=True)
    box_neuron = grid_neuron.clip_box(bounds=[0, 3000e-7, 0, 3000e-7, 0, 5000e-7], invert=True)
    roi_box = pyvista.Box(bounds=(x_L, x_U, y_L, y_U, z_L, z_U))

    # Plot the original (ghosted) and the slice
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(box_neuron, color=c_neuron)
    p.add_mesh(box_ECS, scalar_bar_args=sargs, color=c_ECS)
    p.add_mesh(grid_glial, scalar_bar_args=sargs_glial, cmap=cmap_glial, clim=[-81, -80.62467193603516])
    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5)

    # add title to colorbar
    p.add_text(
        r"$\phi_M (mV)$",
        position=(0.83, 0.45),      # Right side, halfway up
        orientation=-270,          # Rotate 90 degrees clockwise
        font_size=11,
        color="black",
        viewport=True              # Uses the 0-1 coordinate system
    )

    # Fix camera position and zoom
    p.camera_position = 'yz'
    p.camera.azimuth += 225
    p.camera.elevation += 15
    p.reset_camera()

    # Save screenshot
    p.screenshot(f"results/astrocyte_potential_ECS_embedding.png", transparent_background=True)
    p.close()

def plot_astrocyte_potential(grid_glial, i):

    roi_box = pyvista.Box(bounds=(x_L, x_U, y_L, y_U, z_L, z_U))

    # Plot the original (ghosted) and the slice
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(grid_glial, scalar_bar_args=sargs_glial, cmap=cmap_glial, clim=[-81, -80.62467193603516])
    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5)

    # add title to colorbar
    p.add_text(
        r"$\phi_M (mV)$",
        position=(0.83, 0.45),      # Right side, halfway up
        orientation=-270,          # Rotate 90 degrees clockwise
        font_size=11,
        color="black",
        viewport=True              # Uses the 0-1 coordinate system
    )

    # Fix camera position and zoom
    p.camera_position = 'yz'
    p.camera.azimuth += 225
    p.camera.elevation += 15
    p.reset_camera()

    # Save screenshot
    p.screenshot(f"results/astrocyte_potential.png", transparent_background=True)
    p.close()



i = 1
for time in [92.09999999999904]:
    time_index = 460
    grid_ECS = get_grid_field("results_sub_0", "c_K_0", time_index)
    grid_neuron = get_grid_field("results_sub_1", "c_K_1", time_index)
    grid_glial = get_grid_field("results_mem_2", "phi_M_2", time_index)

    grid_ECS_mesh = get_grid_mesh("results_sub_0", "c_K_0")
    grid_neuron_mesh = get_grid_mesh("results_sub_1", "c_K_1")
    grid_glial_mesh = get_grid_mesh("results_sub_2", "c_K_2")

    plot_astrocyte_potential_ECS_embedding(grid_ECS, grid_neuron, grid_glial, i)
    plot_ECS_K(grid_ECS, i)
    plot_astrocyte_potential(grid_glial, i)

    i += 1
