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
italy = gpd.read_file(path_files_gis / "italy_WGS1984.shp")
fishnet = gpd.read_file(path_files_gis / "fishnet_italy_25km.shp").reset_index().rename(
    columns={"index": "GRID_OID"})
routes = gpd.read_file(path_files_gis / "routes_distances.shp")

# Load separate node files
nodes_cement = gpd.read_file(path_files_gis / "node_italy_cement.shp")
nodes_other = gpd.read_file(path_files_gis / "node_italy_other.shp")
nodes_refinery = gpd.read_file(path_files_gis / "node_italy_refinery.shp")
nodes_waste = gpd.read_file(path_files_gis / "node_italy_wte.shp")
nodes_storage = gpd.read_file(path_files_gis / "node_italy_storage.shp")
nodes_transport = gpd.read_file(path_files_gis / "node_italy_transport.shp")
nodes_selected = gpd.read_file(path_files_gis / "nodes_italy_14.shp")

# Print diagnostic information
print("Node counts by type:")
print(f"Cement: {len(nodes_cement)}")
print(f"Other: {len(nodes_other)}")
print(f"Refinery: {len(nodes_refinery)}")
print(f"Waste: {len(nodes_waste)}")
print(f"Storage: {len(nodes_storage)}")
print(f"Transport: {len(nodes_transport)}")
print(f"Selected: {len(nodes_selected)}")

# Print column names for debugging
print("\nColumn names in each dataset:")
if len(nodes_cement) > 0:
    print(f"Cement columns: {list(nodes_cement.columns)}")
if len(nodes_other) > 0:
    print(f"Other columns: {list(nodes_other.columns)}")
if len(nodes_refinery) > 0:
    print(f"Refinery columns: {list(nodes_refinery.columns)}")
if len(nodes_waste) > 0:
    print(f"Waste columns: {list(nodes_waste.columns)}")
if len(nodes_storage) > 0:
    print(f"Storage columns: {list(nodes_storage.columns)}")
if len(nodes_transport) > 0:
    print(f"Transport columns: {list(nodes_transport.columns)}")
if len(nodes_selected) > 0:
    print(f"Selected columns: {list(nodes_selected.columns)}")

# Load selected nodes early to get the selected names
print(f"Selected nodes loaded: {len(nodes_selected)}")
if 'Name' in nodes_selected.columns:
    selected_names = set(nodes_selected['Name'].dropna().unique())
    print(f"Selected facility names: {list(selected_names)}")
else:
    selected_names = set()
    print("Warning: 'Name' column not found in nodes_selected")

# Add type column and selection status to each dataframe
nodes_cement['Type'] = 'Cement'
if len(nodes_cement) > 0 and 'Facility_N' in nodes_cement.columns:
    nodes_cement['is_selected'] = nodes_cement['Facility_N'].isin(selected_names)
    print(f"Cement nodes selected: {nodes_cement['is_selected'].sum()}")
else:
    nodes_cement['is_selected'] = False

nodes_other['Type'] = 'Other'
if len(nodes_other) > 0 and 'Name' in nodes_other.columns:
    nodes_other['is_selected'] = nodes_other['Name'].isin(selected_names)
    print(f"Other nodes selected: {nodes_other['is_selected'].sum()}")
else:
    nodes_other['is_selected'] = False

nodes_refinery['Type'] = 'Refinery'
if len(nodes_refinery) > 0 and 'Facility_N' in nodes_refinery.columns:
    nodes_refinery['is_selected'] = nodes_refinery['Facility_N'].isin(selected_names)
    print(f"Refinery nodes selected: {nodes_refinery['is_selected'].sum()}")
else:
    nodes_refinery['is_selected'] = False

nodes_waste['Type'] = 'Waste'
if len(nodes_waste) > 0 and 'Facility_N' in nodes_waste.columns:
    nodes_waste['is_selected'] = nodes_waste['Facility_N'].isin(selected_names)
    print(f"Waste nodes selected: {nodes_waste['is_selected'].sum()}")
else:
    nodes_waste['is_selected'] = False

nodes_storage['Type'] = 'Storage'
if len(nodes_storage) > 0 and 'Name' in nodes_storage.columns:
    nodes_storage['is_selected'] = nodes_storage['Name'].isin(selected_names)
    print(f"Storage nodes selected: {nodes_storage['is_selected'].sum()}")
else:
    nodes_storage['is_selected'] = False

nodes_transport['Type'] = 'Transport'
if len(nodes_transport) > 0 and 'Name' in nodes_transport.columns:
    nodes_transport['is_selected'] = nodes_transport['Name'].isin(selected_names)
    print(f"Transport nodes selected: {nodes_transport['is_selected'].sum()}")
else:
    nodes_transport['is_selected'] = False

# Combine all node dataframes (including waste nodes)
nodes_list = [nodes_cement, nodes_other, nodes_refinery, nodes_waste, nodes_storage, nodes_transport]
# Filter out any empty dataframes
nodes_list = [df for df in nodes_list if len(df) > 0]
nodes = pd.concat(nodes_list, ignore_index=True)

print(f"\nTotal nodes after combining: {len(nodes)}")
print(f"Node types in combined data: {nodes['Type'].value_counts().to_dict()}")
print(f"Total selected nodes: {nodes['is_selected'].sum()}")

# Show detailed breakdown of selected nodes by type
selected_by_type = nodes[nodes['is_selected']]['Type'].value_counts()
if len(selected_by_type) > 0:
    print(f"Selected nodes by type: {selected_by_type.to_dict()}")

    # Show the actual names of selected nodes
    print("\nSelected nodes details:")
    for node_type in selected_by_type.index:
        type_selected = nodes[(nodes['Type'] == node_type) & (nodes['is_selected'])]
        if node_type in ['Cement', 'Refinery', 'Waste']:
            name_col = 'Facility_N'
        else:  # Other, Storage, Transport
            name_col = 'Name'

        if name_col in type_selected.columns:
            names = type_selected[name_col].tolist()
            print(f"  {node_type}: {names}")
else:
    print("No nodes were marked as selected")

soil_data = pd.read_csv(path_files_grids / "soil_type_grids_italy.csv")
anthro_data = pd.read_csv(path_files_grids / "anthropisation_grids_italy.csv")
morpho_data = pd.read_csv(path_files_grids / "morphological_feature_grids_italy.csv")

# Process data
print(f"\nColumns in nodes: {list(nodes.columns)}")
if 'Annual_Flu' in nodes.columns:
    print(f"Annual_Flu column found. Non-null values: {nodes['Annual_Flu'].notna().sum()}")
    nodes['Annual_Flu'] = pd.to_numeric(nodes['Annual_Flu'], errors='coerce')
    nodes['Annual_Flu_kton'] = nodes['Annual_Flu'] / 1e6
    print(f"Annual_Flu range: {nodes['Annual_Flu_kton'].min()} to {nodes['Annual_Flu_kton'].max()}")
else:
    print("Annual_Flu column not found - will use fallback plotting")

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
print(f"\nNodes bounds: {nodes_bounds}")
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

# Check if any nodes are outside the northern Italy box
nodes_in_box = gpd.clip(nodes, northern_italy)
print(f"Nodes after clipping to Northern Italy: {len(nodes_in_box)}")
print(f"Nodes by type after clipping: {nodes_in_box['Type'].value_counts().to_dict()}")

# Use the clipped nodes for plotting
nodes = nodes_in_box

# Verify selected nodes are still selected after clipping
total_selected_after_clip = nodes['is_selected'].sum()
print(f"Selected nodes after clipping to Northern Italy: {total_selected_after_clip}")

if total_selected_after_clip != nodes['is_selected'].sum():
    print("Warning: Some selected nodes were outside the Northern Italy region")

# Calculate global min and max for consistent color scale across all factors
min_val = min(fishnet_clipped['MORPH_FACTOR'].min(),
              fishnet_clipped['SOIL_FACTOR'].min(),
              fishnet_clipped['ANTHRO_FACTOR'].min(),
              fishnet_clipped['COST_FACTOR'].min())
max_val = max(fishnet_clipped['MORPH_FACTOR'].max(),
              fishnet_clipped['SOIL_FACTOR'].max(),
              fishnet_clipped['ANTHRO_FACTOR'].max(),
              fishnet_clipped['COST_FACTOR'].max())
print(f"\nGlobal cost factor range: {min_val:.3f} to {max_val:.3f}")

# Create figure with more space for legends
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot cost factor for Northern Italy using cmc.navia_r with global normalization
norm = Normalize(vmin=min_val, vmax=max_val)
fishnet_northern.plot(column='COST_FACTOR', ax=ax, cmap=cmc.navia_r, norm=norm, alpha=0.7, legend=False)

# Plot Northern Italy boundary
italy_northern.boundary.plot(ax=ax, color='black', linewidth=0.8)

# Set equal aspect ratio to prevent oval distortion of circular markers
ax.set_aspect('equal')

# Define updated colors for different node types with more distinguishable greys
node_colors = {
    'Waste': '#CCCCCC',  # Light grey (lighter than before)
    'Wate': '#CCCCCC',  # Same light grey (handling typo)
    'Cement': '#000000',  # Black
    'Other': '#444444',  # Much darker grey (more contrast)
    'Refinery': '#888888',  # Medium grey (more distinct from others)
    'Transport': '#FFFFFF',  # White for transport
    'Storage': '#43A047'  # Green
}

# Create categorization based on Annual_Flu in kton/year
if 'Annual_Flu' in nodes.columns:
    # Fill NaN values with 0 for storage and transport nodes
    nodes['Annual_Flu_kton'] = nodes['Annual_Flu_kton'].fillna(0)

    print(f"\nAnnual_Flu_kton statistics:")
    print(nodes['Annual_Flu_kton'].describe())

    # Create flux categories - handle zero and positive values separately
    # First, create categories for positive flux values
    positive_flux_mask = nodes['Annual_Flu_kton'] > 0
    nodes['category'] = 'Storage/Transport'  # Default for all nodes

    # Apply categories only to nodes with positive flux
    if positive_flux_mask.any():
        positive_categories = pd.cut(
            nodes.loc[positive_flux_mask, 'Annual_Flu_kton'],
            bins=[0, 100, 300, 500, 700, 1000, np.inf],
            labels=['Emitter (0-100)', 'Emitter (100-300)', 'Emitter (300-500)', 'Emitter (500-700)',
                    'Emitter (700-1000)', 'Emitter (>1000)'],
            right=False,
            include_lowest=False
        )
        nodes.loc[positive_flux_mask, 'category'] = positive_categories

    # Ensure Storage and Transport types are always categorized as Storage/Transport regardless of flux
    storage_transport_mask = nodes['Type'].isin(['Storage', 'Transport'])
    nodes.loc[storage_transport_mask, 'category'] = 'Storage/Transport'

    print(f"\nNodes by category: {nodes['category'].value_counts().to_dict()}")

    # Define marker sizes based on flux categories
    size_dict = {
        'Storage/Transport': 150,
        'Emitter (0-100)': 200,
        'Emitter (100-300)': 250,
        'Emitter (300-500)': 400,
        'Emitter (500-700)': 600,
        'Emitter (700-1000)': 800,
        'Emitter (>1000)': 1000
    }

    # Add size column
    nodes['marker_size'] = nodes['category'].map(size_dict)

    # Check for any unmapped categories
    unmapped = nodes[nodes['marker_size'].isna()]
    if len(unmapped) > 0:
        print(f"Warning: {len(unmapped)} nodes have unmapped categories:")
        print(unmapped[['Type', 'category', 'Annual_Flu_kton']].to_string())
        # Assign default size for unmapped nodes
        nodes['marker_size'] = nodes['marker_size'].fillna(150)

    # Calculate proper scaling for circular markers
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    scale_factor = min(x_range, y_range) / 1200

    # Plot each node with custom marker in data coordinates
    nodes_plotted = 0
    for idx, row in nodes.iterrows():
        x, y = row.geometry.x, row.geometry.y
        node_type = row.get('Type', 'Other')
        marker_size = row['marker_size']
        radius = np.sqrt(marker_size) * scale_factor

        # Determine edge color based on selection status
        edge_color = 'red' if row['is_selected'] else 'black'
        edge_width = 2.0 if row['is_selected'] else 1.5

        if node_type == 'Wate and Cement':
            # Split-colored circle for nodes with both Waste and Cement
            circle = Circle((x, y), radius, ec=edge_color, fill=False, lw=edge_width, zorder=5,
                            transform=ax.transData)
            ax.add_patch(circle)

            left_half = patches.Wedge((x, y), radius, 90, 270,
                                      fc=node_colors['Waste'], ec='none', zorder=4,
                                      transform=ax.transData)
            ax.add_patch(left_half)

            right_half = patches.Wedge((x, y), radius, 270, 450,
                                       fc=node_colors['Cement'], ec='none', zorder=4,
                                       transform=ax.transData)
            ax.add_patch(right_half)

        elif node_type == 'Storage':
            # Square marker for Storage
            square_size = radius * 1.8
            rect = Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size,
                             fc=node_colors['Storage'], ec=edge_color, lw=edge_width, zorder=5,
                             transform=ax.transData)
            ax.add_patch(rect)

        elif node_type == 'Transport':
            # Round marker with white fill for Transport
            circle = Circle((x, y), radius, fc=node_colors['Transport'], ec=edge_color, lw=edge_width, zorder=5,
                            transform=ax.transData)
            ax.add_patch(circle)

        else:
            # Standard colored markers for other types (including Refinery)
            color = node_colors.get(node_type, '#888888')
            circle = Circle((x, y), radius, fc=color, ec=edge_color, lw=edge_width, zorder=5,
                            transform=ax.transData)
            ax.add_patch(circle)

        nodes_plotted += 1

    print(f"\nNodes plotted: {nodes_plotted} out of {len(nodes)} total nodes")

    # Create custom legend elements
    legend_elements = []

    # Node type legend items (only include types that exist in the data)
    if len(nodes[nodes['Type'] == 'Waste']) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=node_colors['Waste'],
                                          markeredgecolor='black', markersize=10, label='Waste Emitter'))

    if len(nodes[nodes['Type'] == 'Cement']) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=node_colors['Cement'],
                                          markeredgecolor='black', markersize=10, label='Cement Emitter'))

    if len(nodes[nodes['Type'] == 'Refinery']) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=node_colors['Refinery'],
                                          markeredgecolor='black', markersize=10, label='Refinery Emitter'))

    if len(nodes[nodes['Type'] == 'Other']) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=node_colors['Other'],
                                          markeredgecolor='black', markersize=10, label='Other Emitter'))

    if len(nodes[nodes['Type'] == 'Transport']) > 0:
        transport_circle = plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=node_colors['Transport'],
                                      markeredgecolor='black', markersize=10,
                                      label='Potential Transport Switch')
        legend_elements.append(transport_circle)

    if len(nodes[nodes['Type'] == 'Storage']) > 0:
        storage_square = plt.Line2D([0], [0], marker='s', color='w',
                                    markerfacecolor=node_colors['Storage'],
                                    markeredgecolor='black', markersize=10,
                                    label='Storage Site')
        legend_elements.append(storage_square)

    # Add selected node indicator to legend
    if nodes['is_selected'].any():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='lightgray',
                                          markeredgecolor='red', markeredgewidth=2, markersize=10,
                                          label='Selected Node'))

    # Add node type legend - positioned at lower right with better spacing to avoid overlap
    legend1 = ax.legend(handles=legend_elements, loc='lower right', title='Node Type',
                        frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9,
                        bbox_to_anchor=(0.98, 0.02), fontsize=9)
    ax.add_artist(legend1)

    # Create size legend elements with same sizes as on map
    size_legend_elements = []

    # Calculate legend marker sizes to match the actual map marker sizes
    # Convert from map radius to matplotlib marker size (points^2)
    points_per_inch = 72.0
    fig_width_inches = fig.get_figwidth()
    fig_height_inches = fig.get_figheight()

    # Get the data coordinate range that the axes spans
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    data_width = xlim[1] - xlim[0]
    data_height = ylim[1] - ylim[0]

    # Get axes position in figure coordinates
    bbox = ax.get_position()
    axes_width_inches = bbox.width * fig_width_inches
    axes_height_inches = bbox.height * fig_height_inches

    # Calculate conversion factor from data coordinates to points
    data_to_points_x = (axes_width_inches * points_per_inch) / data_width
    data_to_points_y = (axes_height_inches * points_per_inch) / data_height
    data_to_points = min(data_to_points_x, data_to_points_y)

    # Create size legend elements with same sizes as on map - only for categories that exist
    # Get actual flux categories present in data (excluding Storage/Transport)
    actual_flux_categories = [cat for cat in nodes['category'].unique() if
                              cat != 'Storage/Transport' and cat.startswith('Emitter')]
    actual_flux_categories.sort()  # Sort for consistent order

    for category in actual_flux_categories:
        # Extract the range from the category name for display
        if '(' in category and ')' in category:
            label = category.split('(')[1].split(')')[0] + ' kton/year'
        else:
            label = category

        marker_size = size_dict[category]
        # Calculate the radius in data coordinates (same as on map)
        radius_data = np.sqrt(marker_size) * scale_factor
        # Convert to points for legend (matplotlib marker size is in points^2)
        radius_points = radius_data * data_to_points
        legend_marker_size = radius_points * 2  # diameter in points (matplotlib uses diameter)

        # Ensure marker size is reasonable for legend (clamp between 4 and 20)
        legend_marker_size = max(4, min(20, legend_marker_size))

        size_legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markeredgecolor='black', markersize=legend_marker_size,
                       label=label)
        )

    # Add size legend - moved to upper left corner
    if size_legend_elements:  # Only add legend if there are elements
        legend2 = ax.legend(handles=size_legend_elements, loc='upper left', title='Annual Flux (kton/year)',
                            frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9,
                            bbox_to_anchor=(0.02, 0.98), labelspacing=1.5, handletextpad=1.2,
                            borderpad=1.0, columnspacing=1.0)
        ax.add_artist(legend2)

else:
    # Fallback if Annual_Flu is not available
    print("Using fallback plotting (no Annual_Flu data)")
    nodes_plotted = 0
    # Plot each node type with different colors
    for node_type, color in node_colors.items():
        type_nodes = nodes[nodes['Type'] == node_type]
        if len(type_nodes) > 0:
            print(f"Plotting {len(type_nodes)} {node_type} nodes")

            # Create separate series for selected and non-selected nodes
            selected_nodes = type_nodes[type_nodes['is_selected']]
            non_selected_nodes = type_nodes[~type_nodes['is_selected']]

            if len(non_selected_nodes) > 0:
                if node_type == 'Storage':
                    non_selected_nodes.plot(ax=ax, color=color, markersize=100, alpha=0.8,
                                            edgecolor='black', marker='s')
                elif node_type == 'Transport':
                    non_selected_nodes.plot(ax=ax, facecolor=color, markersize=100, alpha=0.8,
                                            edgecolor='black', marker='o')
                else:
                    non_selected_nodes.plot(ax=ax, color=color, markersize=100, alpha=0.8,
                                            edgecolor='black')

            if len(selected_nodes) > 0:
                if node_type == 'Storage':
                    selected_nodes.plot(ax=ax, color=color, markersize=100, alpha=0.8,
                                        edgecolor='red', marker='s', linewidth=2)
                elif node_type == 'Transport':
                    selected_nodes.plot(ax=ax, facecolor=color, markersize=100, alpha=0.8,
                                        edgecolor='red', marker='o', linewidth=2)
                else:
                    selected_nodes.plot(ax=ax, color=color, markersize=100, alpha=0.8,
                                        edgecolor='red', linewidth=2)

            nodes_plotted += len(type_nodes)
    print(f"Total nodes plotted in fallback: {nodes_plotted}")

# Add colorbar for cost factor with original global normalization (using min_val and max_val)
cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
sm = plt.cm.ScalarMappable(cmap=cmc.navia_r, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Cost Factor for Unit Cost of Pipeline', fontsize=12)

# Add title
ax.set_title('Northern Italy Potential CCS Nodes Distribution', fontsize=16, pad=20)
ax.set_axis_off()

# Add scale indication - moved to lower left corner
ax.text(0.02, 0.02, '25 km grid cells', transform=ax.transAxes,
        fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
        verticalalignment='bottom')

# Adjust layout
try:
    plt.tight_layout()
except:
    pass  # Ignore tight_layout warnings for complex layouts
plt.subplots_adjust(right=0.9)

plt.savefig('northern_italy_network_improved.png', dpi=600, bbox_inches='tight')
plt.show()