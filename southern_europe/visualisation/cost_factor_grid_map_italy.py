import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import cmcrameri.cm as cmc
import matplotlib as mpl
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


# Load Italy boundary
italy = gpd.read_file("/Users/ykk/Desktop/Thesis/data/italy_WGS1984.shp")

# Load or create fishnet grid
fishnet = gpd.read_file("/Users/ykk/Desktop/Thesis/data/fishnet_italy_25km.shp") \
    .reset_index().rename(columns={"index": "GRID_OID"})

# Load data with soil values
soil_data = pd.read_csv(
    "/Users/ykk/Documents/GitHub/AdOpT-NET0_XiaoY/southern_europe/northern_italy_data/geographical_feature/soil_type_grids_italy.csv")

# Load data with anthropisation values
anthro_data = pd.read_csv(
    "/Users/ykk/Documents/GitHub/AdOpT-NET0_XiaoY/southern_europe/northern_italy_data/geographical_feature/anthropisation_grids_italy.csv")

# Load data with morphological values
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

# ——— Plot 1: three‐panel row for the individual factors ———
fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=False)
plt.subplots_adjust(wspace=0.05, right=0.85)  # Reduce space between subplots and make room for legend

# Find the global min and max for all three factors to create a consistent color scale
min_val = min(fishnet_clipped['MORPH_FACTOR'].min(),
              fishnet_clipped['SOIL_FACTOR'].min(),
              fishnet_clipped['ANTHRO_FACTOR'].min())
max_val = max(fishnet_clipped['MORPH_FACTOR'].max(),
              fishnet_clipped['SOIL_FACTOR'].max(),
              fishnet_clipped['ANTHRO_FACTOR'].max())

# Create a normalized colormap
norm = Normalize(vmin=min_val, vmax=max_val)
cmap = cmc.navia_r  # Using reversed navia colormap for all plots

panel_info = [
    ('MORPH_FACTOR', 'a) Geomorphological feature'),
    ('SOIL_FACTOR', 'b) Soil type'),
    ('ANTHRO_FACTOR', 'c) Anthropisation')
]

# Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Create the plots
for i, (col, title) in enumerate(panel_info):
    ax = axes[i]
    italy.boundary.plot(ax=ax, color='black', linewidth=0.8)

    # Plot with normalized colormap
    fishnet_clipped.plot(column=col, ax=ax, cmap=cmap, norm=norm, legend=False)
    fishnet_clipped.boundary.plot(ax=ax, color='gray', linewidth=0.3, alpha=0.5)

    # Place title at the bottom of the subplot
    ax.set_title(title, y=-0.1, fontsize=12)
    ax.set_axis_off()

# Add a colorbar to the right of the subplots - moved further right
cbar_ax = fig.add_axes([0.87, 0.2, 0.02, 0.6])  # [left, bottom, width, height] - increased left position
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Factor Value', fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.savefig('italy_incremental_cost_factors.png', dpi=600, bbox_inches='tight')
plt.show()

# ——— Plot 2: single map for the total cost factor ———
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
italy.boundary.plot(ax=ax, color='black', linewidth=1)

# Plot without legend first
fishnet_clipped.plot(column='COST_FACTOR', ax=ax, cmap=cmc.navia_r, legend=False)
fishnet_clipped.boundary.plot(ax=ax, color='gray', linewidth=0.3, alpha=0.5)
ax.set_axis_off()

# Add colorbar on the right side with more width
cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])  # [left, bottom, width, height] - wider and on right
sm = plt.cm.ScalarMappable(cmap=cmc.navia_r, norm=plt.Normalize(
    fishnet_clipped['COST_FACTOR'].min(), fishnet_clipped['COST_FACTOR'].max()))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Cost Factor Value', fontsize=12)

# Adjust figure to make room for the colorbar
plt.subplots_adjust(right=0.8)

plt.savefig('italy_cost_factor.png', dpi=600, bbox_inches='tight')
plt.show()
