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

# Region in which to apply the source term (cm)
x_L = 2100.0; x_U = 2900.0
y_L = 2100.0; y_U = 2900.0
z_L = 2100.0; z_U = 2500.0

x_M = 2683.0
y_M = 2889.0
z_M = 2206.0

# center point (c,c,c)
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

def plot_2D(x, origin, camera_position, grid_ECS, grid_neuron, grid_glial, grid_syn_1, grid_syn_2):

    roi_box = pyvista.Box(bounds=(x_L, x_U, y_L, y_U, z_L, z_U))
    #roi_point = pyvista.PolyData([x_M, y_M, z_M])

    slice_plane_ECS = grid_ECS.slice(normal=x, origin=origin)
    slice_plane_neuron = grid_neuron.slice(normal=x, origin=origin)
    slice_plane_glial = grid_glial.slice(normal=x, origin=origin)
    slice_plane_syn_1 = grid_syn_1.slice(normal=x, origin=origin)
    slice_plane_syn_2 = grid_syn_2.slice(normal=x, origin=origin)
    slice_roi_box = roi_box.slice(normal=x, origin=origin)

    if x == 'x':
        slice_ort_ECS = grid_ECS.slice_orthogonal(x=x_M, y=y_M, z=z_M)
        slice_ort_neuron = grid_neuron.slice_orthogonal(x=x_M, y=y_M, z=z_M)
        slice_ort_glial = grid_glial.slice_orthogonal(x=x_M, y=y_M, z=z_M)
        slice_ort_syn_1 = grid_syn_1.slice_orthogonal(x=x_M, y=y_M, z=z_M)
        slice_ort_syn_2 = grid_syn_2.slice_orthogonal(x=x_M, y=y_M, z=z_M)

        # Plot the original (ghosted) and the slice
        p = pyvista.Plotter(off_screen=True)
        p.add_mesh(slice_ort_ECS, color=c_ECS)
        p.add_mesh(slice_ort_neuron, color=c_neuron)
        p.add_mesh(slice_ort_glial, color=c_glial)
        p.add_mesh(slice_ort_syn_1, color=c_synapse_1)
        p.add_mesh(slice_ort_syn_2, color=c_synapse_2)

        # Customizing the axes
        p.add_axes(
            line_width=8,                # Thicker lines
            label_size=(0.15, 0.1),      # Size of the X, Y, Z text
            viewport=(0.05, 0, 0.5, 0.4), # Makes the whole widget much bigger (40% of window)
            color='white',               # Changes the font/label color
        )

        # 4. Save the screenshot
        p.screenshot(f"results/ort.png", transparent_background=True)
        p.close()

    p = pyvista.Plotter(off_screen=True)
    #p.add_mesh(grid, opacity=0.1, color='white') # See-through original
    p.add_mesh(slice_plane_ECS, scalar_bar_args=sargs, color=c_ECS, label="ECS")
    p.add_mesh(slice_plane_neuron, scalar_bar_args=sargs, color=c_neuron, label="neuron")
    p.add_mesh(slice_plane_glial, scalar_bar_args=sargs, color=c_glial, label="glial")
    p.add_mesh(slice_plane_syn_1, scalar_bar_args=sargs, color=c_synapse_1, label="pre synapse")
    if x != 'z':
        p.add_mesh(slice_plane_syn_2, scalar_bar_args=sargs, color=c_synapse_2, label="post synapse")
    p.add_mesh(slice_roi_box, color="black", style="wireframe", line_width=3, label="ROI")
    #p.add_mesh(roi_point, color=c_point, point_size=10, render_points_as_spheres=True)

    if x == 'z':
        legend = p.add_legend(size=(0.15, 0.15), face='circle', loc='upper right')
        # Access the underlying VTK properties for deeper control
        legend.GetEntryTextProperty().SetFontSize(12)
        legend.GetEntryTextProperty().BoldOn()  # Force bolding
        legend.GetEntryTextProperty().SetFontFamilyToCourier()

    # Focus the camera tightly on the object
    p.reset_camera()
    p.camera.zoom(1.0) # Increase zoom to 'crop' out the edges
    p.camera_position = camera_position

    # Customizing the axes
    #p.add_axes(
    #    line_width=8,                # Thicker lines
    #    label_size=(0.15, 0.1),      # Size of the X, Y, Z text
    #    viewport=(0.05, 0, 0.5, 0.4), # Makes the whole widget much bigger (40% of window)
    #    color='white',               # Changes the font/label color
    #)

    # Save the screenshot
    p.screenshot(f"results/img_{x}.png", transparent_background=True)
    p.close()

    # Zoom in
    p = pyvista.Plotter(off_screen=True)
    roi_bounds = [x_L, x_U, y_L, y_U, z_L, z_U]

    clipped_ECS = slice_plane_ECS.clip_box(bounds=roi_bounds, invert=False)
    clipped_glial = slice_plane_glial.clip_box(bounds=roi_bounds, invert=False)
    clipped_neuron = slice_plane_neuron.clip_box(bounds=roi_bounds, invert=False)
    clipped_syn_1 = slice_plane_syn_1.clip_box(bounds=roi_bounds, invert=False)
    clipped_syn_2 = slice_plane_syn_2.clip_box(bounds=roi_bounds, invert=False)

    p.add_mesh(clipped_ECS, color=c_ECS)
    p.add_mesh(clipped_glial, color=c_glial)
    p.add_mesh(clipped_neuron, color=c_neuron)
    p.add_mesh(clipped_syn_1, color=c_synapse_1)
    if x == 'z':
        p.add_mesh(clipped_syn_2, color=c_synapse_2)
    #p.add_mesh(roi_point, color=c_point, point_size=40, render_points_as_spheres=True)

    p.camera_position = camera_position

    # 1. Focus the camera tightly on the object
    p.reset_camera()
    p.camera.zoom(2.0) # Increase zoom to 'crop' out the edges

    # 4. Save the screenshot
    p.screenshot(f"results/2D_mesh_roi_{x}.png", transparent_background=True)
    p.close()

def plot_mesh_overview(grid_ECS, grid_neuron, grid_glial, grid_syn_1, grid_syn_2):

    box_ECS = grid_ECS.clip_box(bounds=[0, 3000, 0, 3000, 0, 5000], invert=True)
    box_neuron = grid_neuron.clip_box(bounds=[0, 3000, 0, 3000, 0, 5000], invert=True)

    roi_box = pyvista.Box(bounds=(x_L, x_U, y_L, y_U, z_L, z_U))

    # Plot the original (ghosted) and the slice
    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(box_neuron, color=c_neuron, label="neuron")
    p.add_mesh(box_ECS, scalar_bar_args=sargs, color=c_ECS, label="ECS")
    p.add_mesh(grid_glial, scalar_bar_args=sargs, color=c_glial, label="glial")
    p.add_mesh(grid_syn_1, scalar_bar_args=sargs, opacity=0.75, color=c_synapse_1, label="post synapse")
    p.add_mesh(grid_syn_2, scalar_bar_args=sargs, opacity=0.75, color=c_synapse_2, label="pre synapse")
    p.add_mesh(roi_box, color="black", style="wireframe", line_width=5, label="ROI")

    # Fix camera position and zoom
    p.camera_position = 'yz'
    #p.camera.up = (0, 0, -1)
    p.camera.azimuth += 225
    #p.camera.azimuth += 135
    p.camera.elevation += 15
    p.reset_camera()
    #p.camera.zoom(1.3) # Increase zoom to 'crop' out the edges

    legend = p.add_legend(size=(0.15, 0.15), face='circle', loc='upper right')
    # Access the underlying VTK properties for deeper control
    legend.GetEntryTextProperty().SetFontSize(15)
    legend.GetEntryTextProperty().BoldOn()  # Force bolding
    legend.GetEntryTextProperty().SetFontFamilyToCourier()

    # Save screenshot
    p.screenshot(f"results/mesh_overview.png", transparent_background=True)
    p.close()

time = 0.1
grid_ECS = get_grid("results_sub_0", "c_K_0", time)
grid_neuron = get_grid("results_sub_1", "c_K_1", time)
grid_glial = get_grid("results_sub_2", "c_K_2", time)
grid_syn_1 = get_grid("results_sub_3", "c_K_3", time)
grid_syn_2 = get_grid("results_sub_4", "c_K_4", time)

#plot_2D('x', [x_M, c, c], "yz", grid_ECS, grid_neuron, grid_glial, grid_syn_1, grid_syn_2)
#plot_2D('y', [c, y_M, c], "xz", grid_ECS, grid_neuron, grid_glial, grid_syn_1, grid_syn_2)
#plot_2D('z', [c, c, z_M], "xy", grid_ECS, grid_neuron, grid_glial, grid_syn_1, grid_syn_2)

plot_mesh_overview(grid_ECS, grid_neuron, grid_glial, grid_syn_1, grid_syn_2)
