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
    position_x=0.80,           # Move left/right (0 to 1)
    position_y=0.27,           # Move up/down (0 to 1)
    width=0.1,                 # Width of the bar
    height=0.5,                # Height of the bar
    title_font_size=27,
    label_font_size=27,
    color='black',
)

sargs_glial = dict(
    title=" ",
    n_labels=6,                # Number of labels
    fmt="%.2f",                # Decimal formatting
    font_family="arial",
    vertical=True,             # Horizontal orientation
    position_x=0.84,           # Move left/right (0 to 1)
    position_y=0.27,           # Move up/down (0 to 1)
    width=0.1,                 # Width of the bar
    height=0.5,                # Height of the bar
    title_font_size=27,
    label_font_size=27,
    color='black',
)

cmap_ECS = "cool"
cmap_glial = "plasma"
#cmap_glial = "cool"

# Region in which to apply the source term (cm)
#x_L = 2100e-7; x_U = 2900e-7
#y_L = 2100e-7; y_U = 2900e-7
#z_L = 2100e-7; z_U = 2500e-7

x_L = 2000e-7; x_U = 3000e-7
y_L = 2000e-7; y_U = 3000e-7
z_L = 2100e-7; z_U = 2700e-7

x_M = 2683e-7
y_M = 2889e-7
z_M = 2206e-7

roi_bounds = [x_L, x_U, y_L, y_U, z_L, z_U]
roi_box = pyvista.Box(bounds=(x_L, x_U, y_L, y_U, z_L, z_U))
roi_point = pyvista.PolyData([x_M, y_M, z_M])

# center point (c,c,c)
c = 2500e-7

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

def get_grid_field(dir, finame, funame, time_index):
    # Read mesh from file
    filename = f"../results/{dir}/{finame}.xdmf"
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

def plot_ECS_K(fname, grid_ECS, grid_ECS_init, clim, text, i):

    slice_plane_ECS = grid_ECS.slice(normal='x', origin=[x_M, y_M, z_M])
    slice_plane_roi = roi_box.slice(normal='x', origin=[x_M, y_M, z_M])

    # Then assign it back to a mesh to plot it
    diff_array = grid_ECS.point_data["c_K_0"] - grid_ECS_init.point_data["c_K_0"]
    grid_ECS["diff"] = diff_array

    slice_plane_ECS = grid_ECS.clip(normal='x')
    slice_plane_roi = roi_box.slice(normal='x')

    # Make full 3D plot
    p = pyvista.Plotter(off_screen=True)

    p.add_mesh(slice_plane_ECS,
               scalars="diff",
               scalar_bar_args=sargs_ECS,
               cmap=cmap_ECS,
               clim=clim
               )

    #p.add_mesh(roi_point, color=c_point, point_size=10, render_points_as_spheres=True)

    p.add_mesh(slice_plane_roi,
               color="black",
               style="wireframe",
               line_width=3,
               show_edges=True)

    # Fix camera position and zoom
    p.camera_position = 'yz'

    # add title to colorbar
    p.add_text(
        r"$\Delta [\rm K]_e (mM)$",
        position=(0.95, 0.44),     # Right side, halfway up
        orientation=-270,          # Rotate 90 degrees clockwise
        font_size=13,
        color="black",
        viewport=True              # Uses the 0-1 coordinate system
    )

    p.add_text(
        text,
        position=(0.45, 0.88),     # Right side, halfway up
        font_size=15,
        color="black",
        viewport=True              # Uses the 0-1 coordinate system
    )

    p.save_graphic(f"results/{fname}.svg")
    p.close()

    """
    # PLot glial potential in roi
    #grid_ECS_roi = grid_ECS.clip_box(bounds=roi_bounds, invert=False)
    grid_ECS_roi = slice_plane_ECS.clip_box(bounds=roi_bounds, invert=False)

    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(
        grid_ECS_roi,
        scalars="diff",
        scalar_bar_args=sargs_ECS, \
        cmap=cmap_ECS, #clim=clim
    )
    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5)
    p.add_mesh(roi_point, color=c_point, point_size=20, render_points_as_spheres=True)
    # Fix camera position and zoom
    p.camera_position = 'xz'
    p.camera.azimuth += 40
    p.camera.elevation += 200
    # Save screenshot
    # This strips out the solid canvas layer before exporting
    p.save_graphic(f"results/{fname}_roi.svg")
    p.close()
    """


def plot_ECS_and_glial(fname, grid_ECS, grid_ECS_init, grid_glial, grid_glial_init, clim, text, i):

    #slice_plane_ECS = grid_ECS.slice(normal='x')
    #slice_plane_roi = roi_box.slice(normal='x')

    # Then assign it back to a mesh to plot it
    diff_array = grid_ECS.point_data["c_K_0"] - grid_ECS_init.point_data["c_K_0"]
    grid_ECS["diff"] = diff_array

    # Then assign it back to a mesh to plot it
    diff_array = grid_glial.point_data["phi_M_2"] - grid_glial_init.point_data["phi_M_2"]
    grid_glial["diff"] = diff_array

    # Plot the original (ghosted) and the slice
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(
        grid_glial,
        scalars="diff",
        scalar_bar_args=sargs_glial, \
        cmap=cmap_glial, #clim=clim
    )
    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5)

    # add title to colorbar
    p.add_text(
        r"$\Delta \phi_M \rm (mV)$",
        position=(0.98, 0.44),     # Right side, halfway up
        orientation=-270,           # Rotate 90 degrees clockwise
        font_size=13,
        color="black",
        viewport=True              # Uses the 0-1 coordinate system
    )
    # add title to colorbar
    p.add_text(
        text,
        position=(0.5, 0.84),      # Right side, halfway up
        font_size=15,
        color="black",
        viewport=True               # Uses the 0-1 coordinate system
    )

    # Fix camera position and zoom
    p.camera_position = 'xy'
    p.camera.azimuth += 30
    p.camera.elevation += 180
    # Save screenshot
    p.screenshot(f"results/{fname}.png", transparent_background=True)
    p.close()

    # PLot glial potential in roi
    grid_glial_roi = grid_glial.clip_box(bounds=roi_bounds, invert=False)
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(
        grid_glial_roi,
        scalars="diff",
        scalar_bar_args=sargs_glial, \
        cmap=cmap_glial, #clim=clim
    )
    p.add_mesh(roi_point, color=c_point, point_size=10, render_points_as_spheres=True)
    # Fix camera position and zoom
    p.camera_position = 'xy'
    p.camera.azimuth += 30
    p.camera.elevation += 180
    # Save screenshot

    p.screenshot(f"results/{fname}_roi.png", transparent_background=True)
    p.close()



def plot_astrocyte_potential_ECS_embedding(grid_ECS, grid_neuron, grid_glial, i):

    box_ECS = grid_ECS.clip_box(bounds=[0, 3000e-7, 0, 3000e-7, 0, 5000e-7], invert=True)
    box_neuron = grid_neuron.clip_box(bounds=[0, 3000e-7, 0, 3000e-7, 0, 5000e-7], invert=True)

    # Plot the original (ghosted) and the slice
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(box_neuron, color=c_neuron)
    p.add_mesh(box_ECS, scalar_bar_args=sargs, color=c_ECS)
    p.add_mesh(grid_glial, scalar_bar_args=sargs_glial, cmap=cmap_glial, clim=[-81, -80.62467193603516])

    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5)
    #p.add_mesh(roi_point, color=c_point, point_size=10, render_points_as_spheres=True)

    # add title to colorbar
    p.add_text(
        r"$\phi_M (mV)$",
        position=(0.83, 0.45),      # Right side, halfway up
        orientation=-270,           # Rotate 90 degrees clockwise
        font_size=13,
        color="black",
        viewport=True               # Uses the 0-1 coordinate system
    )

    # Fix camera position and zoom
    p.camera_position = 'yz'
    p.camera.azimuth += 225
    p.camera.elevation += 15
    p.reset_camera()

    # Save screenshot
    p.screenshot(f"results/astrocyte_potential_ECS_embedding_{i}.png", transparent_background=True)
    p.close()

def plot_astrocyte_potential(fname, grid_glial, grid_glial_init, clim, text, i):

    # Then assign it back to a mesh to plot it
    diff_array = grid_glial.point_data["phi_M_2"] - grid_glial_init.point_data["phi_M_2"]
    grid_glial["diff"] = diff_array

    # Plot the original (ghosted) and the slice
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(
        grid_glial,
        scalars="diff",
        scalar_bar_args=sargs_glial, \
        cmap=cmap_glial, #clim=clim
    )
    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5)

    # add title to colorbar
    p.add_text(
        r"$\Delta \phi_M \rm (mV)$",
        position=(0.98, 0.44),     # Right side, halfway up
        orientation=-270,           # Rotate 90 degrees clockwise
        font_size=13,
        color="black",
        viewport=True              # Uses the 0-1 coordinate system
    )
    # add title to colorbar
    p.add_text(
        text,
        position=(0.5, 0.84),      # Right side, halfway up
        font_size=15,
        color="black",
        viewport=True               # Uses the 0-1 coordinate system
    )

    # Fix camera position and zoom
    p.camera_position = 'xy'
    p.camera.azimuth += 30
    p.camera.elevation += 180
    # Save screenshot
    p.screenshot(f"results/{fname}.png", transparent_background=True)
    p.close()

    # PLot glial potential in roi
    grid_glial_roi = grid_glial.clip_box(bounds=roi_bounds, invert=False)
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(
        grid_glial_roi,
        scalars="diff",
        scalar_bar_args=sargs_glial, \
        cmap=cmap_glial, clim=clim
    )
    p.add_mesh(roi_point, color=c_point, point_size=10, render_points_as_spheres=True)
    # Fix camera position and zoom
    p.camera_position = 'xy'
    p.camera.azimuth += 30
    p.camera.elevation += 180
    # Save screenshot

    p.screenshot(f"results/{fname}_roi.png", transparent_background=True)
    p.close()

#dir = "baseline"
#text = r"$\rm baseline$"
#clim = [6.88, 9.05] # adjusted ECS
#fname = "astrocyte_potential_bs"

# ICS
#------------------------------------#

#dir = "ICS-tort-x13"
#text = r"$\rm \lambda_i \times 1.3$"
#clim = [6.88, 9.05] # adjusted ECS
#fname = "astrocyte_potential_I13"

#dir = "ICS-tort-x31"
#text = r"$\rm \lambda_i \times 3.1$"
#clim = [6.88, 9.05] # adjusted ECS
#fname = "astrocyte_potential_I31"

#dir = "ICS-tort-x5"
#text = r"$\rm \lambda_i \times 4.4$"
#clim = [6.88, 9.05] # adjusted ECS
#fname = "astrocyte_potential_I44"

# ECS
#------------------------------------#

#dir = "ECS-tort-x13"
#text = r"$\rm \lambda_e \times 1.3$"
#clim = [7.16, 13.02] # adjusted ECS
##fname = "astrocyte_potential_E13"
#fname = "ECS_K_E13"

#dir = "ECS-tort-x31"
#text = r"$\rm \lambda_e \times 3.1$"
#clim = [7.16, 13.02] # adjusted ECS
##fname = "astrocyte_potential_E31"
#fname = "ECS_K_E31"

# ECS-ICS
#------------------------------------#

#dir = "baseline"
#clim = [7.66, 13.02] # adjusted ECS
#text = r"$\rm baseline$"
#fname = "astrocyte_potential_bs_EI"

#dir = "ECS-ICS-tort-x13"
#text = r"$\rm \lambda_e \times 1.3$"
#clim = [7.66, 10.54] # adjusted ECS
#fname = "astrocyte_potential_EI13"

#dir = "ECS-tort-x31"
#text = r"$\rm \lambda_e \times 3.1$"
#clim = [7.66, 10.54] # adjusted ECS
#fname = "astrocyte_potential_EI31"

dir = "ECS-tort-x44"
text = r"$\rm \lambda_e \times 4.4$"
clim = [7.43, 12.89] # adjusted ECS
fname = "astrocyte_potential_E44"

dir = "ECS-ICS-tort-x44"
text = r"$\rm \lambda_e, \lambda_i \times 4.4$"
clim = [7.43, 15.89] # adjusted ECS
fname = "astrocyte_potential_EI44"

#dir = "baseline"
#text = r"$\rm baseline$"
#fname = "astrocyte_potential_bs"
#clim = [7.43, 15.89]    # adjusted ECS
##fname = "ECS_K_bs"
##clim = [0.01, 8.87]     # adjusted ECS

i = 1
index_1 = 184
index_2 = 185
index_3 = 186
#times = [r't = 92.1 ms', r't = 92.6 ms', r't = 93.1 ms']

# Make plots
#for time_index in [index_1, index_2, index_3]:
for time_index in [index_1]:

    fname_i = f"{fname}_{i}"
    #text = times[i-1]

    # -------- plot glial membrane potential ------------- "
    grid_glial = get_grid_field(dir, "results_mem_2", "phi_M_2", time_index)
    grid_glial_init = get_grid_field(dir, "results_mem_2", "phi_M_2", 0)

    # Remove small islands in plot
    ri_grid_glial = grid_glial.connectivity(extraction_mode='largest')
    ri_grid_glial_init = grid_glial_init.connectivity(extraction_mode='largest')
    # Plot membrane potential
    plot_astrocyte_potential(fname_i, ri_grid_glial, ri_grid_glial_init, clim, text, i)

    """
    # -------- plot ECS K+ ------------- "
    grid_ECS = get_grid_field(dir, "results_sub_0", "c_K_0", time_index)
    grid_ECS_init = get_grid_field(dir, "results_sub_0", "c_K_0", 0)
    plot_ECS_K(fname_i, grid_ECS, grid_ECS_init, clim, text, i)

    """
    #plot_ECS_and_glial(fname_i, grid_ECS, grid_ECS_init, grid_glial, grid_glial_init, clim, text, i)

    # -------- div ------------- "
    #grid_neuron = get_grid_field(dir, "results_sub_1", "c_K_1", time_index)
    #grid_ECS_mesh = get_grid_mesh("results_sub_0", "c_K_0")
    #grid_neuron_mesh = get_grid_mesh("results_sub_1", "c_K_1")
    #grid_glial_mesh = get_grid_mesh("results_sub_2", "c_K_2")

    #plot_astrocyte_potential_ECS_embedding(grid_ECS, grid_neuron, grid_glial, i)

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
