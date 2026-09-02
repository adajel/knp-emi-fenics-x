import matplotlib.pyplot as plt
import pandas as pd

# 1. Configure Matplotlib to use LaTeX's Computer Modern font
plt.rcParams.update({
    "text.usetex": False,            
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"]
})

# 2. Define your exact 4 columns and 2 rows
columns = ["", "D1", "D2", "D2"]
data = [
    ["Avg. width ECS (nm)",   "68 (30)"  , "68 (30)"  , "68 (30)"],
    ["Avg. width glial (nm)", "241 (145)", "241 (145)", "241 (145)"]
]
df = pd.DataFrame(data, columns=columns)

# 3. Setup the figure
fig, ax = plt.subplots(figsize=(5.5, 1.2))
ax.axis('tight')
ax.axis('off')

# 4. Create the table with FORCED equal column widths 
# [0.25, 0.25, 0.25, 0.25] splits the table into 4 perfectly equal quarters
table = ax.table(
    cellText=df.values, 
    colLabels=df.columns, 
    cellLoc='left', 
    #colWidths=[0.35, 0.21, 0.21, 0.21], # Gave the first column slightly more room for labels
    colWidths=[0.43, 0.21, 0.21, 0.21], # Gave the first column slightly more room for labels
    loc='center'
)

# 5. Apply LaTeX "booktabs" styling with strict Left Padding alignment
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.6) 

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('black')

    # x=0.05 guarantees every heading and data point starts at the exact same left-pixel offset
    if row == 0:
        cell.set_text_props(weight='bold', x=0.05)
        cell.visible_edges = 'TB'
        cell.set_linewidth(1.5)    # \toprule
    elif row == len(df):
        cell.set_text_props(x=0.05)
        cell.visible_edges = 'B'
        cell.set_linewidth(1.5)    # \bottomrule
    else:
        cell.set_text_props(x=0.05)
        if row == 1:
            cell.visible_edges = 'T' # \midrule
            cell.set_linewidth(0.8)
        else:
            cell.visible_edges = ''

# 6. Save directly to SVG
plt.savefig(
    "results/table_avg_width.svg",
    format="svg",
    bbox_inches="tight",
    transparent=True
)

print("Table generated!")
