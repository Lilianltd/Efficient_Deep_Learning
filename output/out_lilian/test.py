import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np

# Data
csv_content = """Model (Run),Trainable Parameters,Value (%),Score
DenseNet121,"6,956,298",95.12,2.222081779245606
ResNet18,"11,173,962",94.16,2
MobileNetV2_custom_0.5_1,"602,482",91.46,0.09802929719892478
MobileNetV2,"2,296,922",91.07,0.3697387035892479
MobileNetV2_custom_0.75_0.75,"1,017,190",91.06,0.15666128018263129
MobileNetV2_custom_0.5_0.5,"332,018",88.97,0.05092638450265266
MobileNetV2_custom_0.5_0.75,"465,906",90.28,0.07216707295869083
MobileNetV2_custom_0.5_1_pruned,"602,482",90.95,0.06396309362862902
DenseNet121_0.85_4b,"6,956,298",91.17,0.42
DenseNet121_0.35_struc_0.7un_4b,"6,956,298",92,0.27"""

df = pd.read_csv(io.StringIO(csv_content))
df['Trainable Parameters'] = df['Trainable Parameters'].astype(str).str.replace(',', '')
df['Trainable Parameters'] = pd.to_numeric(df['Trainable Parameters'], errors='coerce')
df['Params (M)'] = df['Trainable Parameters'] / 1_000_000

def smart_label(ax, x, y, labels, x_is_log=False):
    x = np.array(x)
    y = np.array(y)
    
    # Filter out NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    labels = np.array(labels)[mask]
    
    if len(x) == 0:
        return

    # Use log scale coordinates if necessary for distance calculation
    x_plot = np.log10(x) if x_is_log else x
    
    # Normalize for fair distance calculation
    x_min, x_max = x_plot.min(), x_plot.max()
    y_min, y_max = y.min(), y.max()
    
    # Avoid division by zero
    x_range = (x_max - x_min) if x_max != x_min else 1
    y_range = (y_max - y_min) if y_max != y_min else 1
    
    xn = (x_plot - x_min) / x_range
    yn = (y - y_min) / y_range
    
    points = np.column_stack((xn, yn))
    
    for i, (xi, yi, lbl) in enumerate(zip(x, y, labels)):
        # Calculate vector away from the nearest neighbor
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        nearest_idx = np.argmin(dists)
        
        # Direction vector from nearest neighbor to current point
        dx = points[i, 0] - points[nearest_idx, 0]
        dy = points[i, 1] - points[nearest_idx, 1]
        
        # If points are identical, default to top-right
        if dx == 0 and dy == 0:
            dx, dy = 1, 1
        else:
            mag = np.sqrt(dx**2 + dy**2)
            dx, dy = dx/mag, dy/mag
            
        # Determine offset and alignment
        # Scale displacement based on a fixed pixel-like distance
        offset_base = 30
        ox, oy = dx * offset_base, dy * offset_base
        
        # Tweak alignments to prevent the box from being on top of the point
        ha = 'left' if dx > 0 else 'right'
        va = 'bottom' if dy > 0 else 'top'
        
        ax.annotate(lbl, (xi, yi), 
                    xytext=(ox, oy), textcoords='offset points',
                    ha=ha, va=va, fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.7, lw=0.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.6, connectionstyle="arc3,rad=0.1"))

# Plot 1: Acc vs Million Params (Log X)
plt.figure(figsize=(12, 7))
plot_df1 = df.dropna(subset=['Params (M)', 'Value (%)']).sort_values('Params (M)')
plt.scatter(plot_df1['Params (M)'], plot_df1['Value (%)'], color='royalblue', s=120, edgecolors='black', zorder=5, alpha=0.8)
plt.xscale('log')
plt.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% Accuracy Threshold')

smart_label(plt.gca(), plot_df1['Params (M)'], plot_df1['Value (%)'], plot_df1['Model (Run)'], x_is_log=True)

plt.xlabel('Trainable Parameters (Millions, Log Scale)', fontweight='bold')
plt.ylabel('Accuracy (%)', fontweight='bold')
plt.title('Performance vs Model Size (Complexity)', fontsize=14, fontweight='bold', pad=15)
plt.grid(True, which="both", linestyle='--', alpha=0.4)
plt.legend(frameon=True, loc='lower right')
plt.tight_layout()
plt.savefig('smart_acc_vs_params.png', dpi=120)
plt.close()

# Plot 2: Acc vs Score (Log X)
plt.figure(figsize=(12, 7))
plot_df2 = df.dropna(subset=['Score', 'Value (%)']).sort_values('Score')
plt.scatter(plot_df2['Score'], plot_df2['Value (%)'], color='seagreen', s=120, edgecolors='black', zorder=5, alpha=0.8)
plt.xscale('log')
plt.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% Accuracy Threshold')

smart_label(plt.gca(), plot_df2['Score'], plot_df2['Value (%)'], plot_df2['Model (Run)'], x_is_log=True)

plt.xlabel('Efficiency Score (Log Scale)', fontweight='bold')
plt.ylabel('Accuracy (%)', fontweight='bold')
plt.title('Performance vs Efficiency Score', fontsize=14, fontweight='bold', pad=15)
plt.grid(True, which="both", linestyle='--', alpha=0.4)
plt.legend(frameon=True, loc='lower right')
plt.tight_layout()
plt.savefig('smart_acc_vs_score.png', dpi=120)
plt.close()

print("Enhanced smart labels generated using vector-based repulsion logic.")