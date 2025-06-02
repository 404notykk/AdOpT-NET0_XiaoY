import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import cmcrameri.cm as cmc
from matplotlib.colors import Normalize
from shapely.geometry import box
import numpy as np
from matplotlib.patches import Patch

# Load data
italy = gpd.read_file("/Users/ykk/Desktop/Thesis/data/italy_WGS1984.shp")
fishnet = gpd.read_file("/Users/ykk/Desktop/Thesis/data/fishnet_italy_25km.shp").reset_index().rename(
    columns={"index": "GRID_OID"})
routes = gpd.read_file("/Users/ykk/Desktop/Thesis/data/routes_distances.shp")
nodes = gpd.read_file("/Users/ykk/Desktop/Thesis/data/nodes_italy_14.shp")

# Print nodes information to troubleshoot
print(f"Number of nodes: {len(nodes)}")
print(f"Nodes CRS: {nodes.crs}")
print(f"Nodes columns: {nodes.columns.tolist()}")

# Check for Annual_Flu column
if 'Annual_Flu' in nodes.columns:
    print(f"Annual_Flu column found")
    # Convert to numeric if needed
    nodes['Annual_Flu'] = pd.to_numeric(nodes['Annual_Flu'], errors='coerce')
    print(f"Annual_Flu range: {nodes['Annual_Flu'].min()} to {nodes['Annual_Flu'].max()}")
    print(f"Annual_Flu unique values: {nodes['Annual_Flu'].unique()}")

# Load attribute data and merge
soil_data = pd.read_csv(
    "/Users/ykk/Documents/GitHub/AdOpT-NET0_XiaoY/southern_europe/northern_italy_data/geographical_feature/soil_type_grids_italy.csv")
anthro_data = pd.read_csv(
    "/Users/ykk/Documents/GitHub/AdOpT-NET0_XiaoY/southern_europe/northern_italy_data/geographical_feature/anthropisation_grids_italy.csv")
morpho_data = pd.read_csv(
    "/Users/ykk/Documents/GitHub/AdOpT-NET0_XiaoY/southern_europe/northern_italy_data/geographical_feature/morphological_feature_grids_italy.csv")

# Merge all attributes into fishnet
fishnet = (fishnet
           .merge(soil_data, on="GRID_OID")
           .merge(anthro_data, on="GRID_OID")
           .merge(morpho_data, on="GRID_OID"))

# Calculate factors
fishnet['SOIL_FACTOR'] = 0.025 * fishnet['NON_ROCK_S'] + 0.21 * fishnet['ROCK_S']
fishnet['ANTHRO_FACTOR'] = 0.0025 * fishnet['NON_ANTHROPISED_A'] + 0.38 * fishnet['ANTHROPISED_A']
fishnet['MORPH_FACTOR'] = 0.025 * fishnet['PLAIN_M'] + 0.06 * fishnet['HILL_M'] + 0.09 * fishnet['MOUNTAIN_M']
fishnet['COST_FACTOR'] = fishnet[['SOIL_FACTOR', 'ANTHRO_FACTOR', 'MORPH_FACTOR']].sum(axis=1)

# Clip to Italy boundary
fishnet_clipped = gpd.clip(fishnet, italy)

# Create a bounding box for Northern Italy based on nodes distribution
nodes_bounds = nodes.total_bounds  # (minx, miny, maxx, maxy)
buffer_size = 0.5  # degrees
northern_italy_box = box(
    nodes_bounds[0] - buffer_size,
    nodes_bounds[1] - buffer_size,
    nodes_bounds[2] + buffer_size,
    nodes_bounds[3] + buffer_size
)

# Convert to GeoDataFrame with same CRS as other data
northern_italy = gpd.GeoDataFrame(geometry=[northern_italy_box], crs=italy.crs)

# Clip data to Northern Italy
fishnet_northern = gpd.clip(fishnet_clipped, northern_italy)
italy_northern = gpd.clip(italy, northern_italy)

# Create figure and plot cost factor
fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# Plot cost factor for Northern Italy
fishnet_northern.plot(column='COST_FACTOR', ax=ax, cmap=cmc.navia_r, alpha=0.7, legend=False)

# Plot Northern Italy boundary
italy_northern.boundary.plot(ax=ax, color='black', linewidth=0.8)

# Plot routes
routes.plot(ax=ax, color='gray', linewidth=1.5, alpha=0.7)

# Prepare nodes for plotting using the specified Annual_Flu categories
if 'Annual_Flu' in nodes.columns:
    # Ensure Annual_Flu is numeric
    nodes['Annual_Flu'] = pd.to_numeric(nodes['Annual_Flu'], errors='coerce')

    # Print the distribution to understand the actual values
    print(f"Annual_Flu distribution: \n{nodes['Annual_Flu'].describe()}")

    # Create categorization using the specific thresholds
    nodes['category'] = 'Storage/Transport'  # Default category

    # Use the specific thresholds you provided
    nodes.loc[(nodes['Annual_Flu'] > 0) & (nodes['Annual_Flu'] <= 300000000), 'category'] = 'Emitter (100-300M)'
    nodes.loc[(nodes['Annual_Flu'] > 300000000) & (nodes['Annual_Flu'] <= 500000000), 'category'] = 'Emitter (300-500M)'
    nodes.loc[(nodes['Annual_Flu'] > 500000000) & (nodes['Annual_Flu'] <= 700000000), 'category'] = 'Emitter (500-700M)'
    nodes.loc[
        (nodes['Annual_Flu'] > 700000000) & (nodes['Annual_Flu'] <= 1000000000), 'category'] = 'Emitter (700-1000M)'

    # Print the counts for each category
    print(nodes['category'].value_counts())

    # Define plotting properties for each category
    category_props = {
        'Storage/Transport': {'color': 'blue', 'marker': 's', 'size': 80},
        'Emitter (100-300M)': {'color': 'green', 'marker': 'o', 'size': 100},
        'Emitter (300-500M)': {'color': 'yellow', 'marker': 'o', 'size': 150},
        'Emitter (500-700M)': {'color': 'orange', 'marker': 'o', 'size': 200},
        'Emitter (700-1000M)': {'color': 'red', 'marker': 'o', 'size': 250}
    }

    # Plot each category
    for category, props in category_props.items():
        category_nodes = nodes[nodes['category'] == category]
        if not category_nodes.empty:
            category_nodes.plot(
                ax=ax,
                color=props['color'],
                marker=props['marker'],
                markersize=props['size'],
                alpha=0.8,
                label=category,
                edgecolor='black',
                zorder=5
            )
            print(f"Plotted {len(category_nodes)} nodes in category: {category}")
else:
    # Fallback if Annual_Flu is not available
    print("Annual_Flu column not found - plotting nodes with default style")
    nodes.plot(ax=ax, color='red', markersize=100, alpha=0.8, edgecolor='black')

# Add colorbar for cost factor
cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
sm = plt.cm.ScalarMappable(cmap=cmc.navia_r, norm=Normalize(
    fishnet_northern['COST_FACTOR'].min(), fishnet_northern['COST_FACTOR'].max()))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Cost Factor', fontsize=12)

# Adjust layout
plt.subplots_adjust(right=0.8)

# Add title
ax.set_title('Northern Italy CO₂ Transport Network', fontsize=16)
ax.set_axis_off()

# Add legend
ax.legend(loc='lower right', frameon=True,
          facecolor='white', edgecolor='gray', framealpha=0.8, fontsize=10)

# Add scale indication
ax.text(0.02, 0.02, '25 km grid cells', transform=ax.transAxes,
        fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

plt.savefig('northern_italy_network_fixed.png', dpi=600, bbox_inches='tight')
plt.show()
