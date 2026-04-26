import matplotlib.pyplot as plt
import numpy as np

c_neuron = "#16a085"
c_glial = "#ff67ff"
c_synapse_1 = "#00ff00"
c_synapse_2 = "#e1fae1"
c_point = "#ffff00"

# --- Data ---
labels = ['astro', 'neuron']

vol_a = 51.97
vol_n = 838
tot = vol_a + vol_n
values = [vol_a, vol_n]  # um^3

p_a = vol_a/tot
p_n = vol_n/tot

#print(vol_a)
#print(vol_n)

percentages = ['6%', '94%']

# Define colors based on the image
colors = [c_glial, c_neuron]

# --- Create Plot ---
fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars
bars = ax.bar(labels, values, color=colors)

# Add percentage labels above bars
for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 2, pct, 
            ha='center', va='bottom', fontsize=14)

# --- Styling ---
ax.set_ylabel(r'surface area ($\mu \rm{m}^2$)', fontsize=16)
ax.set_ylim(0, 80)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Customize grid (horizontal lines only)
ax.yaxis.grid(True, color='black', linewidth=1.5)
ax.set_axisbelow(False) # Keep grid lines on top of bars

# Formatting ticks
plt.xticks(rotation=30, fontsize=14)
plt.yticks([200, 400, 600, 800, 1000], fontsize=12)
ax.tick_params(direction='out', length=10, width=2)

plt.tight_layout()
plt.savefig("results/surfaces.png")
