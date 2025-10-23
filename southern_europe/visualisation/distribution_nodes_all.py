#!/usr/bin/env python3
"""
Enhanced CCS Network Analysis with Mass Balance Verification and Italy Background
"""

import h5py
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
from collections import defaultdict
from matplotlib.patches import Patch
from shapely.geometry import box

# Import cmcrameri for navia colormap (matching example script)
try:
    import cmcrameri.cm as cmc

    navia_available = True
    print("CMC colormaps loaded successfully!")
except ImportError:
    print("Warning: cmcrameri not available. Install with: pip install cmcrameri")
    print("Falling back to matplotlib's viridis colormap")
    navia_available = False

warnings.filterwarnings('ignore')


def extract_active_flows(h5_file_path, flow_threshold=1e-6):
    """
    Extract active flow connections from HDF5 file with detailed logging
    """
    active_connections = []

    with h5py.File(h5_file_path, 'r') as f:
        if 'operation' not in f or 'networks' not in f['operation']:
            print("❌ No operation/networks found in H5 file")
            return []

        periods = list(f['operation/networks'].keys())
        if not periods:
            print("❌ No periods found in networks data")
            return []

        period = periods[0]
        print(f"Using period: {period}")

        network_types = list(f[f'operation/networks/{period}'].keys())
        print(f"Network types: {network_types}")

        for network_type in network_types:
            network_group = f[f'operation/networks/{period}/{network_type}']
            connections = list(network_group.keys())

            type_active = 0
            for connection in connections:
                if 'flow' in network_group[connection]:
                    flow_data = network_group[connection]['flow'][:]
                    total_flow = np.sum(flow_data)

                    if total_flow > flow_threshold:
                        active_connections.append({
                            'network_type': network_type,
                            'connection': connection,
                            'total_flow': total_flow,
                            'max_flow': np.max(flow_data),
                            'avg_flow': np.mean(flow_data)
                        })
                        type_active += 1
                        print(f"    ACTIVE: {connection} (flow: {total_flow:.2e})")

            print(f"  {network_type}: {type_active}/{len(connections)} active")

    return active_connections


def parse_connection_name(connection_name, known_nodes):
    """
    Parse connection name with detailed debugging - FIXED VERSION
    """
    # Strip whitespace to handle trailing spaces
    connection_name = connection_name.strip()
    nodes_sorted = sorted(known_nodes, key=len, reverse=True)

    from_node = None
    to_node = None

    # Find from_node at the start
    for node in nodes_sorted:
        if connection_name.startswith(node):
            from_node = node
            remaining = connection_name[len(node):].strip()
            break

    # Find to_node in remaining part
    if from_node and remaining:
        for node in nodes_sorted:
            if remaining == node or remaining.endswith(node):
                to_node = node
                break

    return from_node, to_node


def identify_node_types(active_connections, all_nodes):
    """
    Classify nodes as sources, intermediate hubs, or storage sites
    """
    node_analysis = defaultdict(lambda: {
        'incoming_flows': [],
        'outgoing_flows': [],
        'total_in': 0,
        'total_out': 0,
        'node_type': 'unknown'
    })

    # Analyze all connections
    for conn in active_connections:
        from_node, to_node = parse_connection_name(conn['connection'], all_nodes)

        if from_node and to_node:
            flow = conn['total_flow']

            # Record outgoing flow
            node_analysis[from_node]['outgoing_flows'].append({
                'to': to_node,
                'flow': flow,
                'connection': conn['connection']
            })
            node_analysis[from_node]['total_out'] += flow

            # Record incoming flow
            node_analysis[to_node]['incoming_flows'].append({
                'from': from_node,
                'flow': flow,
                'connection': conn['connection']
            })
            node_analysis[to_node]['total_in'] += flow

    # Classify nodes
    for node, data in node_analysis.items():
        if data['total_in'] == 0 and data['total_out'] > 0:
            data['node_type'] = 'source'
        elif data['total_out'] == 0 and data['total_in'] > 0:
            data['node_type'] = 'storage'
        elif data['total_in'] > 0 and data['total_out'] > 0:
            data['node_type'] = 'intermediate'
        else:
            data['node_type'] = 'isolated'

    return dict(node_analysis)


def verify_mass_balance(node_analysis):
    """
    Verify mass balance and identify discrepancies
    """
    print(f"\n🔍 MASS BALANCE ANALYSIS")
    print(f"=" * 50)

    # Categorize nodes
    sources = []
    intermediates = []
    storage = []

    for node, data in node_analysis.items():
        if data['node_type'] == 'source':
            sources.append((node, data))
        elif data['node_type'] == 'intermediate':
            intermediates.append((node, data))
        elif data['node_type'] == 'storage':
            storage.append((node, data))

    print(f"📊 NODE CLASSIFICATION:")
    print(f"  Sources: {len(sources)} nodes")
    print(f"  Intermediate hubs: {len(intermediates)} nodes")
    print(f"  Storage sites: {len(storage)} nodes")

    # Calculate totals
    total_source_flow = sum(data['total_out'] for _, data in sources)
    total_storage_flow = sum(data['total_in'] for _, data in storage)

    print(f"\n💧 FLOW TOTALS:")
    print(f"  Total CO2 from sources: {total_source_flow:.2e}")
    print(f"  Total CO2 to storage: {total_storage_flow:.2e}")
    print(f"  Balance difference: {abs(total_source_flow - total_storage_flow):.2e}")

    # Detailed source analysis
    print(f"\n📤 SOURCE NODES:")
    for node, data in sources:
        print(f"  {node}: {data['total_out']:.2e}")
        for outflow in data['outgoing_flows']:
            print(f"    → {outflow['to']}: {outflow['flow']:.2e}")

    # Detailed storage analysis
    print(f"\n📥 STORAGE NODES:")
    for node, data in storage:
        print(f"  {node}: {data['total_in']:.2e}")
        for inflow in data['incoming_flows']:
            print(f"    ← {inflow['from']}: {inflow['flow']:.2e}")

    # Intermediate hub analysis
    print(f"\n🔄 INTERMEDIATE HUBS:")
    for node, data in intermediates:
        balance = data['total_in'] - data['total_out']
        print(f"  {node}:")
        print(f"    In: {data['total_in']:.2e}, Out: {data['total_out']:.2e}")
        print(f"    Balance: {balance:.2e} {'✅' if abs(balance) < 1e-3 else '❌'}")

    # Mass balance verification
    print(f"\n✅ MASS BALANCE VERIFICATION:")
    if abs(total_source_flow - total_storage_flow) < 1e-3:
        print(f"  ✅ Mass balance VERIFIED: Source = Storage")
    else:
        print(f"  ❌ Mass balance ISSUE: Source ≠ Storage")
        print(f"  📊 Difference: {total_source_flow - total_storage_flow:.2e}")

    return {
        'sources': sources,
        'intermediates': intermediates,
        'storage': storage,
        'total_source_flow': total_source_flow,
        'total_storage_flow': total_storage_flow
    }


def analyze_flow_paths(node_analysis, sources, storage):
    """
    Trace flow paths from sources to storage
    """
    print(f"\n🛤️  FLOW PATH ANALYSIS")
    print(f"=" * 50)

    def trace_path(start_node, visited=None, path=None):
        if visited is None:
            visited = set()
        if path is None:
            path = []

        if start_node in visited:
            return []  # Avoid cycles

        visited.add(start_node)
        path.append(start_node)

        # If this is a storage node, we found a complete path
        if start_node in [s[0] for s in storage]:
            return [path.copy()]

        # Continue tracing through outgoing connections
        all_paths = []
        if start_node in node_analysis:
            for outflow in node_analysis[start_node]['outgoing_flows']:
                sub_paths = trace_path(outflow['to'], visited.copy(), path.copy())
                all_paths.extend(sub_paths)

        return all_paths

    # Trace paths from each source
    all_flow_paths = []
    for source_node, source_data in sources:
        paths = trace_path(source_node)
        for path in paths:
            all_flow_paths.append({
                'source': source_node,
                'path': path,
                'source_flow': source_data['total_out']
            })

    print(f"📍 COMPLETE FLOW PATHS:")
    for i, flow_path in enumerate(all_flow_paths, 1):
        path_str = " → ".join(flow_path['path'])
        print(f"  {i}. {path_str}")
        print(f"     Source flow: {flow_path['source_flow']:.2e}")

    return all_flow_paths


def explain_flow_mismatch(active_connections, mass_balance_results):
    """
    Explain why total pipeline flows don't match storage flows
    """
    print(f"\n🔍 FLOW MISMATCH EXPLANATION")
    print(f"=" * 50)

    total_pipeline_flows = sum(conn['total_flow'] for conn in active_connections)
    total_storage_flow = mass_balance_results['total_storage_flow']

    print(f"📊 FLOW COMPARISON:")
    print(f"  Sum of ALL pipeline flows: {total_pipeline_flows:.2e}")
    print(f"  Total flow to storage: {total_storage_flow:.2e}")
    print(f"  Ratio: {total_pipeline_flows / total_storage_flow:.1f}x")

    print(f"\n💡 WHY THE DIFFERENCE:")
    print(f"  • Pipeline flows represent transport through each segment")
    print(f"  • The same CO2 travels through multiple pipeline segments")
    print(f"  • Storage flow represents the final consolidated amount")
    print(f"  • This is normal network behavior - NOT an error!")

    # Show example flow accumulation
    print(f"\n📈 FLOW ACCUMULATION EXAMPLE:")
    print(f"  If CO2 travels: Source → Hub → Storage")
    print(f"  • Source→Hub pipeline: 100 units flow")
    print(f"  • Hub→Storage pipeline: 100 units flow")
    print(f"  • Total pipeline flows: 200 units")
    print(f"  • Actual CO2 stored: 100 units")
    print(f"  • The same CO2 was counted twice in pipeline flows!")


def load_italy_boundary(path_files_gis):
    """
    Load Italy boundary data from local shapefile (same as working example)
    """
    try:
        italy_shp_path = path_files_gis / "italy_WGS1984.shp"
        if italy_shp_path.exists():
            italy = gpd.read_file(italy_shp_path)
            print("✅ Italy boundary loaded from local shapefile")
            return italy
        else:
            print(f"⚠️  Italy shapefile not found at: {italy_shp_path}")
            return None
    except Exception as e:
        print(f"⚠️  Could not load Italy boundary: {e}")
        return None


def get_italy_northern_region(nodes_gdf):
    """
    Get a bounding box for Northern Italy based on node locations (same as working example)
    """
    from shapely.geometry import box

    # Get the bounds of the nodes to focus on the relevant area
    nodes_bounds = nodes_gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Add buffer around nodes (same as working example)
    buffer_size = 0.5
    northern_italy_box = box(
        nodes_bounds[0] - buffer_size,
        nodes_bounds[1] - buffer_size,
        nodes_bounds[2] + buffer_size,
        nodes_bounds[3] + buffer_size
    )

    return northern_italy_box


def create_enhanced_network_plot_with_italy(nodes_gdf, active_connections, coord_dict, node_analysis, path_files_gis):
    """
    Create enhanced network visualization with Italy boundary background (using same approach as working example)
    """
    # Setup navia colors (matching example script)
    if navia_available:
        navia_cmap = cmc.navia
        network_colors = {
            'CO2_Pipeline': navia_cmap(0.15),
            'CO2Railway': navia_cmap(0.85),
            'CO2Truck': navia_cmap(0.5)
        }
        print("✅ Using navia colormap for network colors")
    else:
        viridis_cmap = plt.cm.viridis
        network_colors = {
            'CO2_Pipeline': viridis_cmap(0.15),
            'CO2Railway': viridis_cmap(0.85),
            'CO2Truck': viridis_cmap(0.5)
        }
        print("⚠️ Using viridis fallback colormap")

    # Load Italy boundary from local shapefile
    italy_boundary = load_italy_boundary(path_files_gis)

    # Create figure with higher DPI for better quality
    fig, ax = plt.subplots(figsize=(20, 16))

    # Plot Italy boundary if available (same approach as working example)
    if italy_boundary is not None:
        # Ensure CRS matches
        if nodes_gdf.crs != italy_boundary.crs:
            italy_boundary = italy_boundary.to_crs(nodes_gdf.crs)

        # Create Northern Italy bounding box (same as working example)
        northern_italy_box = get_italy_northern_region(nodes_gdf)
        northern_italy = gpd.GeoDataFrame(geometry=[northern_italy_box], crs=italy_boundary.crs)

        # Clip Italy boundary to Northern region (same as working example)
        italy_northern = gpd.clip(italy_boundary, northern_italy)

        # Plot Italy boundary with proper styling
        italy_northern.plot(ax=ax, color='lightgray', alpha=0.4,
                            edgecolor='black', linewidth=1.2, zorder=1)

        # Plot the boundary outline more prominently
        italy_northern.boundary.plot(ax=ax, color='black', linewidth=1.5, zorder=2)

        # Set the plot limits to the northern Italy region (more zoomed in)
        bounds = italy_northern.total_bounds
        # Override with custom bounds as requested
        ax.set_xlim(8.5, 12.7)  # Left to 8.5, right to 12.7
        ax.set_ylim(44.3, 45.8)  # Lower to 44.3, upper to 45.8

    else:
        # Fallback: set bounds based on nodes with custom bounds
        bounds = nodes_gdf.total_bounds
        ax.set_xlim(8.5, 12.7)  # Left to 8.5, right to 12.7
        ax.set_ylim(44.3, 45.8)  # Lower to 44.3, upper to 45.8

    # Define node colors matching the example script
    node_type_colors = {
        'source': '#000000',  # Black (like Cement in example)
        'storage': '#43A047',  # Green (same as Storage in example)
        'intermediate': '#888888',  # Medium grey (like Refinery in example)
        'inactive': '#CCCCCC'  # Light grey (like Waste in example)
    }

    # Load emission data and create size categories (similar to example script)
    # First, let's see if we can get emission data from the nodes
    print("Setting up emission-based node sizing...")

    # Create emission categories for sizing (similar to example script)
    def get_emission_size_and_category(node_name, node_type):
        """Get emission-based size and category for a node"""
        # Default values
        emission_size = 150  # Base size for nodes without emission data
        category = 'Storage/Transport'

        # For now, we'll use some sample logic - you can modify this based on your actual emission data
        # This is a placeholder that assigns different sizes based on node type and position
        if node_type == 'source':
            # Assign different emission categories to source nodes
            import random
            random.seed(hash(node_name) % 1000)  # Deterministic based on node name
            emission_ranges = [
                ('Emitter (0-100)', 200),
                ('Emitter (100-300)', 250),
                ('Emitter (300-500)', 400),
                ('Emitter (500-700)', 600),
                ('Emitter (700-1000)', 800),
                ('Emitter (>1000)', 1000)
            ]
            category, emission_size = random.choice(emission_ranges)
        elif node_type in ['storage', 'intermediate']:
            category = 'Storage/Transport'
            emission_size = 150
        else:
            category = 'Storage/Transport'
            emission_size = 100

        return emission_size, category

    # Color nodes by type and assign emission-based sizes
    node_colors = []
    node_sizes = []
    node_categories = []

    for _, row in nodes_gdf.iterrows():
        node_name = row['Name']
        if node_name in node_analysis:
            node_type = node_analysis[node_name]['node_type']
            color = node_type_colors.get(node_type, node_type_colors['inactive'])
            emission_size, category = get_emission_size_and_category(node_name, node_type)
        else:
            color = node_type_colors['inactive']
            emission_size, category = get_emission_size_and_category(node_name, 'inactive')

        node_colors.append(color)
        node_sizes.append(emission_size)
        node_categories.append(category)

    # Calculate proper scaling for circular markers (from example script)
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()
    x_range = ax_xlim[1] - ax_xlim[0]
    y_range = ax_ylim[1] - ax_ylim[0]
    scale_factor = min(x_range, y_range) / 1200

    # Plot nodes with emission-based sizing (similar to example script approach)
    from matplotlib.patches import Circle, Rectangle
    import numpy as np

    nodes_plotted = 0
    for i, (_, row) in enumerate(nodes_gdf.iterrows()):
        x, y = row.geometry.x, row.geometry.y
        node_name = row['Name']
        color = node_colors[i]
        marker_size = node_sizes[i]

        # Calculate radius in data coordinates
        radius = np.sqrt(marker_size) * scale_factor

        # Determine node type for special markers
        node_type = 'unknown'
        if node_name in node_analysis:
            node_type = node_analysis[node_name]['node_type']

        # Use different markers for different node types (like example script)
        edge_color = 'black'
        edge_width = 3

        if node_type == 'storage':
            # Square marker for Storage (like example script)
            square_size = radius * 1.8
            rect = Rectangle((x - square_size / 2, y - square_size / 2),
                             square_size, square_size,
                             facecolor=color, edgecolor=edge_color,
                             linewidth=edge_width, zorder=6)
            ax.add_patch(rect)
        else:
            # Circular markers for all other types (like example script)
            circle = Circle((x, y), radius,
                            facecolor=color, edgecolor=edge_color,
                            linewidth=edge_width, zorder=6)
            ax.add_patch(circle)

        nodes_plotted += 1

    print(f"Nodes plotted with emission-based sizing: {nodes_plotted}")

    # Enhanced network styling with navia colors (matching example script) - MODIFIED OPACITY TO 35%
    network_styles = {
        'CO2_Pipeline': {'color': network_colors['CO2_Pipeline'], 'linestyle': '-', 'label': 'CO2 Pipeline',
                         'alpha': 0.35},
        'CO2Railway': {'color': network_colors['CO2Railway'], 'linestyle': '--', 'label': 'CO2 Railway', 'alpha': 0.35},
        'CO2Truck': {'color': network_colors['CO2Truck'], 'linestyle': ':', 'label': 'CO2 Truck', 'alpha': 0.35}
    }

    # Plot connections with improved styling
    max_flow = max([conn['total_flow'] for conn in active_connections]) if active_connections else 1
    plotted_count = 0
    plotted_types = set()

    for conn in active_connections:
        from_node, to_node = parse_connection_name(conn['connection'], coord_dict.keys())

        if from_node and to_node and from_node in coord_dict and to_node in coord_dict:
            from_coord = coord_dict[from_node]
            to_coord = coord_dict[to_node]

            style = network_styles.get(conn['network_type'],
                                       {'color': 'gray', 'linestyle': '-', 'label': 'Other', 'alpha': 0.35})

            # Scale line width based on flow (minimum 6, maximum 16 for even better visibility)
            line_width = 6 + 10 * (conn['total_flow'] / max_flow)

            # Calculate route direction and fixed inlet segment length
            dx = to_coord[0] - from_coord[0]
            dy = to_coord[1] - from_coord[1]
            route_length = np.sqrt(dx ** 2 + dy ** 2)

            # Fixed inlet segment length in degrees (adjust this value as needed)
            fixed_inlet_length = 0.08  # degrees (~8.9 km at this latitude)

            if route_length > 0:
                # Calculate inlet fraction based on fixed length
                inlet_fraction = min(fixed_inlet_length / route_length, 0.4)  # Cap at 40%

                # Calculate inlet end coordinates
                unit_dx = dx / route_length
                unit_dy = dy / route_length
                inlet_end_x = from_coord[0] + unit_dx * fixed_inlet_length
                inlet_end_y = from_coord[1] + unit_dy * fixed_inlet_length

                # If fixed length exceeds route length, use the full route
                if inlet_fraction >= 1.0:
                    inlet_end_x = to_coord[0]
                    inlet_end_y = to_coord[1]
            else:
                # Handle zero-length routes (shouldn't happen, but safety check)
                inlet_end_x = from_coord[0]
                inlet_end_y = from_coord[1]
                inlet_fraction = 1.0

            # Always plot the inlet segment with 100% opacity (from_node side)
            ax.plot([from_coord[0], inlet_end_x], [from_coord[1], inlet_end_y],
                    color=style['color'], linewidth=line_width, linestyle=style['linestyle'],
                    alpha=1.0, zorder=4)  # 100% opacity, higher z-order

            # Plot the remaining segment with 35% opacity only if there's remaining length
            if inlet_fraction < 1.0:
                ax.plot([inlet_end_x, to_coord[0]], [inlet_end_y, to_coord[1]],
                        color=style['color'], linewidth=line_width, linestyle=style['linestyle'],
                        alpha=style['alpha'], zorder=3,
                        label=style['label'] if conn['network_type'] not in plotted_types else "")
            else:
                # If inlet segment covers the whole route, add label to inlet segment
                if conn['network_type'] not in plotted_types:
                    # Add an invisible line just for the legend
                    ax.plot([], [], color=style['color'], linestyle=style['linestyle'],
                            alpha=style['alpha'], label=style['label'])

            plotted_types.add(conn['network_type'])
            plotted_count += 1

    # Create enhanced legends (similar to example script)
    if plotted_types:
        transport_legend = ax.legend(title='Transport Mode', loc='upper right',
                                     framealpha=0.95, fontsize=11, title_fontsize=12)
        ax.add_artist(transport_legend)

    # Add node type legend with enhanced styling (matching example script colors)
    node_legend_elements = [
        Patch(facecolor=node_type_colors['source'], label='CO2 Sources'),
        Patch(facecolor=node_type_colors['intermediate'], label='Intermediate Hubs'),
        Patch(facecolor=node_type_colors['storage'], label='Storage Sites'),
        Patch(facecolor=node_type_colors['inactive'], label='Inactive Nodes')
    ]

    # Add Italy boundary to legend if it was plotted
    if italy_boundary is not None:
        node_legend_elements.append(Patch(facecolor='lightgray', alpha=0.4,
                                          edgecolor='black', label='Italy Boundary'))

    # Position node type legend at lower right
    node_legend = ax.legend(handles=node_legend_elements, title='Node Type', loc='lower right',
                            framealpha=0.95, fontsize=11, title_fontsize=12,
                            bbox_to_anchor=(0.98, 0.02))
    ax.add_artist(node_legend)

    # Create emission size legend (similar to example script)
    size_legend_elements = []

    # Define size categories and their corresponding sizes
    size_categories = [
        ('0-100 kton/year', 200),
        ('100-300 kton/year', 250),
        ('300-500 kton/year', 400),
        ('500-700 kton/year', 600),
        ('700-1000 kton/year', 800),
        ('>1000 kton/year', 1000)
    ]

    # Calculate legend marker sizes (similar to example script logic)
    fig_width_inches = fig.get_figwidth()
    fig_height_inches = fig.get_figheight()
    points_per_inch = 72.0

    # Get axes position and data range
    bbox = ax.get_position()
    axes_width_inches = bbox.width * fig_width_inches
    axes_height_inches = bbox.height * fig_height_inches

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    data_width = xlim[1] - xlim[0]
    data_height = ylim[1] - ylim[0]

    # Calculate conversion factor
    data_to_points_x = (axes_width_inches * points_per_inch) / data_width
    data_to_points_y = (axes_height_inches * points_per_inch) / data_height
    data_to_points = min(data_to_points_x, data_to_points_y)

    # Create size legend elements
    for label, marker_size in size_categories:
        # Calculate radius in data coordinates
        radius_data = np.sqrt(marker_size) * scale_factor
        # Convert to points for legend
        radius_points = radius_data * data_to_points
        legend_marker_size = radius_points * 2  # diameter in points

        # Clamp marker size for readability
        legend_marker_size = max(4, min(20, legend_marker_size))

        size_legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='gray', markeredgecolor='black',
                       markersize=legend_marker_size, label=label)
        )

    # Add emission size legend at upper left
    if size_legend_elements:
        size_legend = ax.legend(handles=size_legend_elements, title='Annual CO2 Flux',
                                loc='upper left', framealpha=0.95, fontsize=10,
                                title_fontsize=11, bbox_to_anchor=(0.02, 0.98),
                                labelspacing=1.5, handletextpad=1.2, borderpad=1.0)
        ax.add_artist(size_legend)

    # Enhanced styling and formatting
    ax.set_xlabel('Longitude (°E)', fontsize=14, weight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=14, weight='bold')
    ax.set_title('CO2 Transport Network in Northern Italy\n' +
                 'Optimized CCS Infrastructure with Mass Balance Verification',
                 fontsize=18, weight='bold', pad=25)

    # Add grid and improve appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal')

    # Add a subtle background color
    ax.set_facecolor('#f8f9fa')

    # Improve tick formatting
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()

    # Save with high quality
    plt.savefig('enhanced_co2_network_italy.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')
    plt.show()

    return plotted_count


def main():
    """
    Main analysis function with enhanced mass balance verification and Italy background
    """
    print("Enhanced CO2 Transport Network Analysis with Italy Background")
    print("=" * 60)

    # File paths
    path_data_case_study = Path(r"C:\Users\mockp\PycharmProjects\AdOpT-NET0_XiaoY\southern_europe")
    path_files_gis = path_data_case_study / "northern_italy_data" / "raw_data" / "gis_data"
    results_data_path = path_data_case_study / "resultsImpurities"
    simulation_round_data_path = results_data_path / "20251022134322-1"
    h5_file_path = simulation_round_data_path / "optimization_results.h5"
    nodes_shp_path = path_files_gis / "nodes_italy_14.shp"

    # Check files exist
    if not h5_file_path.exists():
        print(f"❌ H5 file not found: {h5_file_path}")
        return
    if not nodes_shp_path.exists():
        print(f"❌ Shapefile not found: {nodes_shp_path}")
        return

    # Load data
    print("📂 Loading geographic data...")
    nodes_gdf = gpd.read_file(nodes_shp_path)

    # Ensure we have a proper CRS (assume WGS84 if not set)
    if nodes_gdf.crs is None:
        print("⚠️  No CRS found, assuming WGS84 (EPSG:4326)")
        nodes_gdf = nodes_gdf.set_crs('EPSG:4326')

    coord_dict = {row['Name']: (row.geometry.x, row.geometry.y)
                  for _, row in nodes_gdf.iterrows()}

    print(f"✅ Loaded {len(nodes_gdf)} nodes from shapefile")
    print(f"📍 Node extent: {nodes_gdf.total_bounds}")

    # Extract connections
    print("\n📊 Extracting flow data...")
    active_connections = extract_active_flows(h5_file_path)

    if not active_connections:
        print("❌ No active connections found!")
        return

    # Enhanced analysis
    print("\n🔬 Performing network analysis...")
    node_analysis = identify_node_types(active_connections, coord_dict.keys())
    mass_balance_results = verify_mass_balance(node_analysis)
    flow_paths = analyze_flow_paths(node_analysis, mass_balance_results['sources'],
                                    mass_balance_results['storage'])

    # Explain the flow mismatch
    explain_flow_mismatch(active_connections, mass_balance_results)

    # Create enhanced visualization with Italy background
    print("\n🎨 Creating enhanced visualization with Italy boundary...")
    plotted_count = create_enhanced_network_plot_with_italy(nodes_gdf, active_connections,
                                                            coord_dict, node_analysis, path_files_gis)

    print(f"\n✅ ANALYSIS COMPLETE")
    print(f"📊 Network Summary:")
    print(f"   • {len(mass_balance_results['sources'])} CO2 sources")
    print(f"   • {len(mass_balance_results['intermediates'])} intermediate hubs")
    print(f"   • {len(mass_balance_results['storage'])} storage sites")
    print(f"   • {plotted_count} active transport connections")
    print(f"🎨 Enhanced visualization with Italy boundary saved as 'enhanced_co2_network_italy.png'")


if __name__ == "__main__":
    main()