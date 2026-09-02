import os
import matplotlib.pyplot as plt
import numpy as np

c_D1 = "#6086BF"
c_D2 = "#796BC6"
c_D3 = "#AE69BF"

# --- Data ---
labels = [r'D1', r'D2', r'D3']

vol_a = 5.2
vol_n = 4
vol_e = 4.5
values = [vol_e, vol_a, vol_n]
percentages = ['5.2', '4', '4.5']

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
ax.set_ylabel(r'$\rm K^{+}$ ions injected (mM)', fontsize=20)

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
plt.yticks([1, 3, 5], fontsize=15)
ax.tick_params(direction='out', length=10, width=2)

plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/barplot_K_injection.png")
