import os
import matplotlib.pyplot as plt
import numpy as np

c_D1 = "#4e5f70"
c_D2 = "#63798e"
c_D3 = "#7993ad"
#c_D3 = "#7DAFA3"

# --- Data ---
labels = [r'D1', r'D2', r'D3']

vol_D1 = 0.052
vol_D2 = 0.082
vol_D3 = 0.0
values = [vol_D1, vol_D2, vol_D3]
percentages = ['0.052', '0.082', '0.0']

colors = [c_D1, c_D2, c_D3]

# --- Create Plot ---
fig, ax = plt.subplots(figsize=(5, 6))

bar_width = 0.25

# Calculate exact flush positions (left edge aligned)
x = np.arange(len(labels)) * bar_width

# Align='edge' guarantees bars start exactly where the previous ends
bars = ax.bar(x, values, color=colors, width=bar_width, align='edge', edgecolor='none')

# Add percentage labels above bars
for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.001, pct,
            ha='center', va='bottom', fontsize=16)

# --- Styling ---
ax.set_ylabel(r'ECS ROI volume ($\mu \rm{m}^3$)', fontsize=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Set x-limits directly to the outer edges of the combined block
total_width = len(labels) * bar_width
padding = 0.05  # Margin on left and right side of the entire block
ax.set_xlim(-padding, total_width + padding)

# Customize grid
ax.yaxis.grid(True, color='black', linewidth=1.5)
ax.set_axisbelow(False)

# Place ticks directly in the center of each edge-aligned bar
tick_positions = x + (bar_width / 2)
plt.xticks(tick_positions, labels, fontsize=15)
plt.yticks([0.03, 0.06, 0.09], fontsize=15)
ax.tick_params(direction='out', length=10, width=2)

plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/barplot_ECS_ROI_volume.png")
