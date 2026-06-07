import os
import matplotlib.pyplot as plt
import numpy as np

# --- 9 Unique Colors (3 shades per category) ---
ecs_colors = ["#4e5f70", "#63798e", "#7993ad"]     
astro_colors = ["#ff67ff", "#df40df", "#b919b9"]   
neuron_colors = ["#16a085", "#1abc9c", "#2ecc71"]  

colors_p1 = [ecs_colors[0], astro_colors[0], neuron_colors[0]]
colors_p2 = [ecs_colors[1], astro_colors[1], neuron_colors[1]]
colors_p3 = [ecs_colors[2], astro_colors[2], neuron_colors[2]]

# --- Data ---
labels = ['ECS', 'astro', 'neuron']

pillar_1 = [24.25, 11.99, 72.86]  
pillar_2 = [21.10, 15.40, 66.20]  
pillar_3 = [27.80, 9.15,  79.50]  

# --- Spacing Controls ---
group_spacing = 0.75  # Lower this number to bring the 3 categories closer together
width = 0.22          # Width of each individual bar

# Calculate center positions for each category group
x = np.arange(len(labels)) * group_spacing  

# --- Create Plot ---
# Reduced figure width from 10 to 7.5 to match the tighter grouping
fig, ax = plt.subplots(figsize=(7.5, 6))

# Plot the three pillars
rects1 = ax.bar(x - width, pillar_1, width, color=colors_p1)
rects2 = ax.bar(x,         pillar_2, width, color=colors_p2)
rects3 = ax.bar(x + width, pillar_3, width, color=colors_p3)

# --- Helper function with NEW Text Formatting ---
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()

        # Format string options: 
        # For integer: f'{int(height)}'
        # For unit suffix: f'{height:.1f} u³'
        label_text = f'{height:.1f}'
        label_text = f'{round(height)}'

        ax.text(
            rect.get_x() + rect.get_width()/2, 
            height + 2.0,       # Slightly increased padding above the bar
            label_text, 
            ha='center', 
            va='bottom', 
            fontsize=15,        # Slightly smaller to fit clustered layout
            #fontweight='bold',  # Made text bold
            color='#2c3e50'     # Clean dark slate color instead of default black
        )

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

# --- Styling ---
ax.set_ylabel(r'volume fraction ($\%$)', fontsize=20)

# Dynamic X-limits to keep the tight groups centered beautifully
ax.set_xlim(min(x) - 2*width, max(x) + 2*width)

# Set category labels at the new condensed center positions
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, fontsize=15)

# Clean up borders
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)

# Customize grid (horizontal lines only, kept on top of bars)
ax.yaxis.grid(True, color='black', linewidth=1.5)
ax.set_axisbelow(False) 

# Formatting ticks
plt.yticks([20, 40, 60, 80], fontsize=15)
ax.tick_params(direction='out', length=10, width=2)

plt.tight_layout()

# Safely create directory and save
os.makedirs("results", exist_ok=True)
plt.savefig("results/volumes_grouped.svg")
