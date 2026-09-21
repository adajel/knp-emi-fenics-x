#!/bin/bash
# Figure 1
# Plot meshes (glial, synpase, ECS) for Figure 1
python3 plot_mesh.py -c baseline_D1
python3 plot_mesh.py -c baseline_D2
python3 plot_mesh.py -c baseline_D3
# Calculate local width (glial, ECS)
python3 compute_local_width.py -c baseline_D1
python3 compute_local_width.py -c baseline_D2
python3 compute_local_width.py -c baseline_D3
# Plot local width (glial, ECS) and print global and ROI avg. for Figure 1
python3 plot_local_width.py -c baseline_D1
python3 plot_local_width.py -c baseline_D2
python3 plot_local_width.py -c baseline_D3
python3 table_avg_width.py
# Make barplots for SVR and volume fractions for Figure 1
python3 barplot_volume_fractions.py
python3 SVR.py
# Figure 2
# Plot glial potential and ECS K+ concentrations for Figure 2
python3 plot_fields.py -c baseline_D1
python3 plot_fields.py -c baseline_D2
python3 plot_fields.py -c baseline_D3
# Traces of glial potential and ECS K+ concentrations in points for Figure 2
python3 plot_3D.py -c baseline_D1 baseline_D2 baseline_D3

