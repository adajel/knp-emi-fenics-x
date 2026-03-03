import matplotlib.pyplot as plt
import numpy as np

c_ECS = "#4e5f70"
c_neuron = "#16a085"
c_glial = "#ff67ff"
c_synapse_1 = "#00ff00"
c_synapse_2 = "#e1fae1"
c_point = "#ffff00"

# --- Data ---
labels = ['ECS', 'astro', 'neuron']

vol_a = 11.99
vol_n = 72.86
vol_e = 24.25
tot = vol_e + vol_a + vol_n
values = [vol_e, vol_a, vol_n]  # um^3

p_e = vol_e/tot
p_a = vol_a/tot
p_n = vol_n/tot

print(p_e)
print(p_a)
print(p_n)

percentages = ['22%', '11%', '67%']

# Define colors based on the image
colors = [c_ECS, c_glial, c_neuron]

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
ax.set_ylabel(r'volume ($\mu \rm{m}^3$)', fontsize=16)
ax.set_ylim(0, 80)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Customize grid (horizontal lines only)
ax.yaxis.grid(True, color='black', linewidth=1.5)
ax.set_axisbelow(False) # Keep grid lines on top of bars

# Formatting ticks
plt.xticks(rotation=30, fontsize=14)
plt.yticks([20, 40, 60, 80], fontsize=12)
ax.tick_params(direction='out', length=10, width=2)

plt.tight_layout()
plt.savefig("volumes.png")
