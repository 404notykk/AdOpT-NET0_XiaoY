import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import cmcrameri.cm as cmc
from matplotlib.colors import Normalize
from shapely.geometry import box
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch, Circle, Rectangle
import matplotlib.path as mpath
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms

# Load data
path_data_case_study = Path("../northern_italy_data")

path_files_gis = path_data_case_study / "raw_data/gis_data"
path_files_grids = path_data_case_study / "geographical_feature"
italy = gpd.read_file(path_files_gis/"italy_WGS1984.shp")
fishnet = gpd.read_file(path_files_gis/"fishnet_italy_25km.shp").reset_index().rename(
    columns={"index": "GRID_OID"})
routes = gpd.read_file(path_files_gis/"routes_distances.shp")
nodes = gpd.read_file(path_files_gis/"nodes_italy_14.shp")
soil_data = pd.read_csv(path_files_grids / "soil_type_grids_italy.csv")
anthro_data = pd.read_csv(path_files_grids / "anthropisation_grids_italy.csv")
morpho_data = pd.read_csv(path_files_grids / "morphological_feature_grids_italy.csv")

# Process data
if 'Annual_Flu' in nodes.columns:
    nodes['Annual_Flu'] = pd.to_numeric(nodes['Annual_Flu'], errors='coerce')
    nodes['Annual_Flu_kton'] = nodes['Annual_Flu'] / 1e6

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

# Create bounding box for Northern Italy based on nodes distribution
nodes_bounds = nodes.total_bounds
buffer_size = 0.5
northern_italy_box = box(
    nodes_bounds[0] - buffer_size,
    nodes_bounds[1] - buffer_size,
    nodes_bounds[2] + buffer_size,
    nodes_bounds[3] + buffer_size
)

# Convert to GeoDataFrame with same CRS
northern_italy = gpd.GeoDataFrame(geometry=[northern_italy_box], crs=italy.crs)

# Clip data to Northern Italy
fishnet_northern = gpd.clip(fishnet_clipped, northern_italy)
italy_northern = gpd.clip(italy, northern_italy)

# Create figure with more space for legends - increased width
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot cost factor for Northern Italy
fishnet_northern.plot(column='COST_FACTOR', ax=ax, cmap=cmc.grayC_r, alpha=0.7, legend=False)

# Plot Northern Italy boundary
italy_northern.boundary.plot(ax=ax, color='black', linewidth=0.8)

# Plot routes
routes.plot(ax=ax, color='gray', linewidth=1.5, alpha=0.7)

# Define colors for different types using navia colormap
navia_colors = cmc.navia(np.linspace(0, 1, 5))  # Get 5 colors from navia colormap
type_colors = {
    'Waste': navia_colors[2],  # First color from navia
    'Wate': navia_colors[2],  # Same as Waste (handling typo)
    'Cement': navia_colors[3],  # Third color from navia
    'Other': navia_colors[4],  # Fifth color from navia
    'Transport': 'none',
    'Storage': navia_colors[0]  # Changed to first color from navia
}

# Create categorization based on Annual_Flu in kton/year
if 'Annual_Flu' in nodes.columns:
    # Create flux categories
    nodes['category'] = pd.cut(
        nodes['Annual_Flu_kton'],
        bins=[0, 100, 300, 500, 700, 1000],
        labels=['Storage/Transport', 'Emitter (100-300)', 'Emitter (300-500)', 'Emitter (500-700)',
                'Emitter (700-1000)'],
        right=True
    )

    # For nodes with Annual_Flu = 0, categorize as Storage/Transport
    nodes.loc[nodes['Annual_Flu_kton'] <= 0, 'category'] = 'Storage/Transport'

    # Define marker sizes based on flux categories - adjusted for better visibility
    size_dict = {
        'Storage/Transport': 150,  # Base size for non-emitters
        'Emitter (100-300)': 250,  # Small emitters
        'Emitter (300-500)': 400,  # Medium emitters
        'Emitter (500-700)': 600,  # Large emitters
        'Emitter (700-1000)': 800  # Largest emitters
    }

    # Add size column
    nodes['marker_size'] = nodes['category'].map(size_dict)

    # Calculate proper scaling for circular markers
    # Get the data range for proper scaling
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

    # Use the smaller range to ensure circles fit well - adjusted scaling
    scale_factor = min(x_range, y_range) / 1200  # Adjusted for better visibility

    # Plot each node with custom marker in data coordinates with proper scaling
    for idx, row in nodes.iterrows():
        x, y = row.geometry.x, row.geometry.y
        node_type = row.get('Type', 'Other')
        marker_size = row['marker_size']

        # Scale radius based on data coordinate system
        radius = np.sqrt(marker_size) * scale_factor

        if node_type == 'Wate and Cement':
            # Create a split-colored circle for nodes with both Waste and Cement (left-right split)
            circle = Circle((x, y), radius, ec='black', fill=False, lw=1.5, zorder=5,
                            transform=ax.transData)
            ax.add_patch(circle)

            # Left half - Waste color
            left_half = patches.Wedge((x, y), radius, 90, 270,
                                      fc=type_colors['Waste'], ec='none', zorder=4,
                                      transform=ax.transData)
            ax.add_patch(left_half)

            # Right half - Cement color
            right_half = patches.Wedge((x, y), radius, 270, 450,
                                       fc=type_colors['Cement'], ec='none', zorder=4,
                                       transform=ax.transData)
            ax.add_patch(right_half)

        elif node_type == 'Storage':
            # Square marker for Storage
            square_size = radius * 1.8
            rect = Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size,
                             fc=type_colors['Storage'], ec='black', lw=1.5, zorder=5,
                             transform=ax.transData)
            ax.add_patch(rect)

        elif node_type == 'Transport':
            # Round marker with no fill for Transport
            circle = Circle((x, y), radius, fc='none', ec='black', lw=1.5, zorder=5,
                            transform=ax.transData)
            ax.add_patch(circle)

        else:
            # Standard colored markers for other types
            color = type_colors.get(node_type, '#888888')  # Default gray if type not found
            circle = Circle((x, y), radius, fc=color, ec='black', lw=1.5, zorder=5,
                            transform=ax.transData)
            ax.add_patch(circle)

    # Create custom legend elements with colored circles for emitters
    legend_elements = []

    # Node type legend items - colored circles for emitters
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=type_colors['Waste'],
                                      markeredgecolor='black', markersize=10, label='Waste Emitter'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=type_colors['Cement'],
                                      markeredgecolor='black', markersize=10, label='Cement Emitter'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=type_colors['Other'],
                                      markeredgecolor='black', markersize=10, label='Other Emitter'))

    # Create a custom transport circle for the legend (blank with black outline)
    transport_circle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                  markeredgecolor='black', markersize=10,
                                  label='Potential Transport Switch')

    # Create a custom storage square for the legend with first navia color
    storage_square = plt.Line2D([0], [0], marker='s', color='w',
                                markerfacecolor=type_colors['Storage'],
                                markeredgecolor='black', markersize=10,
                                label='Storage Site')

    legend_elements.append(transport_circle)
    legend_elements.append(storage_square)

    # Add node type legend - moved to upper right
    legend1 = ax.legend(handles=legend_elements, loc='upper right', title='Node Type',
                        frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9,
                        bbox_to_anchor=(0.98, 0.98))
    ax.add_artist(legend1)

    # Create size legend elements with sizes that are proportional to the flux categories
    size_legend_elements = []

    # Define the flux categories we want to show in legend (excluding Storage/Transport)
    flux_categories = ['Emitter (100-300)', 'Emitter (300-500)', 'Emitter (500-700)', 'Emitter (700-1000)']
    flux_labels = ['100-300', '300-500', '500-700', '700-1000']

    # Create proportional legend marker sizes based on the size_dict values
    # Use a simple scaling that maintains proportionality
    base_legend_size = 6  # Minimum readable size
    max_legend_size = 16  # Maximum reasonable size for legend

    # Get the range of marker sizes for emitters only
    emitter_sizes = [size_dict[cat] for cat in flux_categories]
    min_emitter_size = min(emitter_sizes)
    max_emitter_size = max(emitter_sizes)

    for category, label in zip(flux_categories, flux_labels):
        # Get the marker size for this category
        marker_size = size_dict[category]

        # Scale proportionally between base_legend_size and max_legend_size
        if max_emitter_size > min_emitter_size:
            normalized_size = (marker_size - min_emitter_size) / (max_emitter_size - min_emitter_size)
            legend_marker_size = base_legend_size + normalized_size * (max_legend_size - base_legend_size)
        else:
            legend_marker_size = base_legend_size

        size_legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markeredgecolor='black', markersize=legend_marker_size,
                       label=label + ' kton/year')
        )

    # Add size legend to lower left corner to avoid overlap
    legend2 = ax.legend(handles=size_legend_elements, loc='lower left', title='Annual Flux (kton/year)',
                        frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9,
                        bbox_to_anchor=(0.02, 0.02), labelspacing=1.5, handletextpad=1.2,
                        borderpad=1.0, columnspacing=1.0)
    ax.add_artist(legend2)

else:
    # Fallback if Annual_Flu is not available
    nodes.plot(ax=ax, color='red', markersize=100, alpha=0.8, edgecolor='black')

# Add colorbar for cost factor - positioned to avoid overlap with map
cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # Moved further right and made narrower
sm = plt.cm.ScalarMappable(cmap=cmc.grayC_r, norm=Normalize(
    fishnet_northern['COST_FACTOR'].min(), fishnet_northern['COST_FACTOR'].max()))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Cost Factor for Unit Cost of Pipeline', fontsize=12)

# Add title
ax.set_title('Northern Italy CO₂ Transport Network', fontsize=16, pad=20)
ax.set_axis_off()

# Add scale indication - moved to upper left to avoid overlap
ax.text(0.02, 0.98, '25 km grid cells', transform=ax.transAxes,
        fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
        verticalalignment='top')

# Adjust layout to prevent clipping
plt.tight_layout()
plt.subplots_adjust(right=0.9)  # Make room for colorbar

plt.savefig('northern_italy_network_improved.png', dpi=600, bbox_inches='tight')
plt.show()