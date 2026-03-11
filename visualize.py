import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_enhanced_heatmap(file, title, output_name):
        # Load the grid data
        data = pd.read_csv(file, header=None)
        
        plt.figure(figsize=(12, 7))
        
        # Create heatmap with labels and better styling
        ax = sns.heatmap(
            data, 
            cmap='magma', 
            cbar_kws={'label': 'Average Traffic Volume'},
            xticklabels=20, # Show every 20th bin label to avoid crowding
            yticklabels=2   # Show every 2nd hour
        )
        
        # Adding titles and labels
        plt.title(title, fontsize=16, pad=20, fontweight='bold')
        plt.xlabel('Weather Index (PC1 Bins: Improved → Worsened)', fontsize=12)
        plt.ylabel('Hour of Day (0-23)', fontsize=12)
        
        # Improve layout to prevent clipping
        plt.tight_layout()
        plt.savefig(output_name, dpi=300)
        plt.close()
        print(f"Successfully created: {output_name}")

# Generate the two plots for your report
plot_enhanced_heatmap('grid_z.csv', 'Traffic Volume Distribution (Original Data)', 'heatmap_original.png')
plot_enhanced_heatmap('smoothed_final.csv', 'Traffic Volume Distribution (2D Gaussian Smoothed)', 'heatmap_smoothed.png')
