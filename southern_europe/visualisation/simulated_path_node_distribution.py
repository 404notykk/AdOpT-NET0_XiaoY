#!/usr/bin/env python3
"""
Visualisation of Simulated Network with Mass Balance Verification with emission data extracted from H5 file for Each Node
"""

import h5py
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from matplotlib.patches import Circle, Rectangle, Patch
from shapely.geometry import box
import cmcrameri.cm as cmc

# ============================================================================
# Setup paths
# ============================================================================
print("=" * 60)
print("CO2 Transport Network Analysis")
print("=" * 60)

script_dir = Path(__file__).parent
path_data_case_study = script_dir.parent / 'italy_data'
path_files_gis = path_data_case_study / 'raw_data' / 'gis_data'
results_data_path = script_dir.parent / 'Results_CCSchainOptimization'
simulation_round_data_path = results_data_path / '20251113121620-1'
h5_file_path = simulation_round_data_path / 'optimization_results.h5'
nodes_shp_path = path_files_gis / 'all_nodes_italy.shp'
italy_shp_path = path_files_gis / "italy_WGS1984.shp"

# ============================================================================
# STEP 1: Load nodes from shapefile
# ============================================================================
print(f"\n{'=' * 60}")
print("STEP 1: Loading nodes...")
print(f"{'=' * 60}")

nodes_gdf = gpd.read_file(nodes_shp_path)
nodes_gdf['Name'] = nodes_gdf['node_name']

# Count unique node locations (some nodes share coordinates)
unique_locations = nodes_gdf.groupby(['geometry']).size()
unique_nodes = nodes_gdf['Name'].nunique()

coord_dict = {row['Name']: (row.geometry.x, row.geometry.y)
              for _, row in nodes_gdf.iterrows()}

print(f"✅ Loaded {len(nodes_gdf)} node records ({unique_nodes} unique names, {len(unique_locations)} unique locations)")
print(f"   Node types: {nodes_gdf['node_type'].value_counts().to_dict()}")

# ============================================================================
# STEP 2: Extract emission data from H5 for each node
# ============================================================================
print(f"\n{'=' * 60}")
print("STEP 2: Extracting emission data...")
print(f"{'=' * 60}")

node_emissions = {}

with h5py.File(h5_file_path, 'r') as f:
    period = list(f['operation/technology_operation'].keys())[0]
    tech_op_group = f[f'operation/technology_operation/{period}']

    for node_name in tech_op_group.keys():
        node_group = tech_op_group[node_name]

        # Each node might have multiple technologies
        for tech_name in node_group.keys():
            tech_group = node_group[tech_name]

            # Extract emission data
            if 'CO2captured_var_output_ccs' in tech_group:
                co2_captured_data = tech_group['CO2captured_var_output_ccs'][:]

                # Sum over time periods (720 time steps)
                total_co2_captured = 36.5 * np.sum(co2_captured_data) # check!

                # Also extract emissions if available
                total_emissions_pos = 0
                if 'emissions_pos' in tech_group:
                    emissions_pos_data = tech_group['emissions_pos'][:]
                    total_emissions_pos = np.sum(emissions_pos_data)

                node_emissions[node_name] = {
                    'co2_captured': total_co2_captured,
                    'emissions_pos': total_emissions_pos,
                    'technology': tech_name
                }

                print(f"  {node_name}")
                print(f"    Technology: {tech_name}")
                print(f"    CO2 Captured: {total_co2_captured:.2f} t/yr")
                print(f"    Emissions Positive: {total_emissions_pos:.2f} t/yr")
# ============================================================================
# STEP 3: Calculate abatement rates
# ============================================================================
print(f"\n{'=' * 60}")
print("STEP 3: Calculating abatement rates...")
print(f"{'=' * 60}")

node_categories = {}

for node_name, emission_data in node_emissions.items():
    co2_captured = emission_data['co2_captured']
    emissions_pos = emission_data['emissions_pos']

    # Total potential CO2 = captured + positive emissions
    total_co2 = co2_captured + emissions_pos

    if total_co2 > 0:
        abatement_rate = co2_captured / total_co2

        # Categorize based on abatement rate
        if abatement_rate >= 0.89:  # ~90% capture (with margin)
            node_categories[node_name] = 'emitter_full'
        elif abatement_rate > 0.01:  # >1% capture
            node_categories[node_name] = 'emitter_partial'
        else:  # ≤1% capture
            node_categories[node_name] = 'emitter_none'

        print(f"  {node_name}: {abatement_rate * 100:.1f}% abatement → {node_categories[node_name]}")
    else:
        node_categories[node_name] = 'emitter_none'
        print(f"  {node_name}: 0% abatement → emitter_none")

# Add storage nodes
with h5py.File(h5_file_path, 'r') as f:
    if 'design/nodes/period1' in f:
        for node_name in f['design/nodes/period1'].keys():
            if node_name not in node_categories:
                # Check if this is a storage node
                node_group = f[f'design/nodes/period1/{node_name}']
                for tech_name in node_group.keys():
                    if 'Storage' in tech_name:
                        node_categories[node_name] = 'storage'
                        print(f"  {node_name}: storage site")
                        break

# ============================================================================
# STEP 4: Extract active flows
# ============================================================================
print(f"\n{'=' * 60}")
print("STEP 4: Extracting flows...")
print(f"{'=' * 60}")

active_connections = []
flow_threshold = 1e-6

with h5py.File(h5_file_path, 'r') as f:
    period = list(f['operation/networks'].keys())[0]
    network_types = list(f[f'operation/networks/{period}'].keys())

    for network_type in network_types:
        network_group = f[f'operation/networks/{period}/{network_type}']
        type_active = 0

        for connection in network_group.keys():
            if 'flow' in network_group[connection]:
                flow_data = network_group[connection]['flow'][:]
                total_flow = np.sum(flow_data)

                if total_flow > flow_threshold:
                    active_connections.append({
                        'network_type': network_type,
                        'connection': connection,
                        'total_flow': total_flow
                    })
                    type_active += 1

        print(f"  {network_type}: {type_active} active connections")

# ============================================================================
# STEP 5: Summary Statistics
# ============================================================================
print(f"\n{'=' * 60}")
print("SUMMARY STATISTICS")
print(f"{'=' * 60}")

emitters_full = [n for n, cat in node_categories.items() if cat == 'emitter_full']
emitters_partial = [n for n, cat in node_categories.items() if cat == 'emitter_partial']
emitters_none = [n for n, cat in node_categories.items() if cat == 'emitter_none']
storage_sites = [n for n, cat in node_categories.items() if cat == 'storage']

print(f"\n📊 NETWORK COMPOSITION:")
print(f"  CO2 Emitters:")
print(f"    • Full Capture (~90%): {len(emitters_full)}")
print(f"    • Partial Capture (0-90%): {len(emitters_partial)}")
print(f"    • No Capture (0%): {len(emitters_none)}")
print(f"  Storage Sites: {len(storage_sites)}")
print(f"  Active Transport Connections: {len(active_connections)}")

# Calculate total emissions
total_captured = sum(node_emissions.get(n, {}).get('co2_captured', 0) for n in node_emissions)
total_vented = sum(node_emissions.get(n, {}).get('emissions_pos', 0) for n in node_emissions)
overall_abatement = total_captured / (total_captured + total_vented) if (total_captured + total_vented) > 0 else 0

print(f"\n📊 OVERALL PERFORMANCE:")
print(f"  Total CO2 Captured: {total_captured:.2f} t/yr")
print(f"  Total CO2 Vented: {total_vented:.2f} t/yr")
print(f"  Network Abatement Rate: {overall_abatement * 100:.1f}%")

print(f"\n{'=' * 60}")
print("✅ ANALYSIS COMPLETE")
print(f"{'=' * 60}")

# ============================================================================
# STEP 6: Enhanced Visualization with Bidirectional Flow Handling
# ============================================================================
print(f"\n{'=' * 60}")
print("STEP 6: Creating enhanced visualization...")
print(f"{'=' * 60}")

# 1. Map results back to GeoDataFrame
active_nodes_in_flow = set()
for conn in active_connections:
    for node_name in nodes_gdf['Name']:
        if node_name in conn['connection']:
            active_nodes_in_flow.add(node_name)


def refine_category(row):
    name = row['Name']
    if name in node_categories:
        return node_categories[name]
    if name in active_nodes_in_flow:
        return 'hub_active'
    return 'hub_inactive'


nodes_gdf['category'] = nodes_gdf.apply(refine_category, axis=1)
nodes_gdf['annual_captured'] = nodes_gdf['Name'].map(
    lambda x: node_emissions.get(x, {}).get('co2_captured', 0)
)

# 2. Setup colors
try:
    navia_cmap = cmc.navia
    network_colors = {
        'CO2_Pipeline': navia_cmap(0.15),
        'CO2Railway': navia_cmap(0.85),
        'CO2Truck': navia_cmap(0.5)
    }
except:
    network_colors = {
        'CO2_Pipeline': '#1f77b4',
        'CO2Railway': '#d62728',
        'CO2Truck': '#7f7f7f'
    }

node_colors = {
    'emitter_full': '#88c879',  # Soft green
    'emitter_partial': '#e2e9ba',  # light green
    'emitter_none': '#D32F2F',  # Red
    'hub_active': '#044977',  # Navy blue
    'hub_inactive': '#BDBDBD',  # Light gray
    'storage': '#041a39',  # Dark blue
    'unknown': '#CCCCCC'
}

# 3. Create Figure with wider aspect ratio
fig, ax = plt.subplots(figsize=(24, 12))

# Plot Italy boundary
if italy_shp_path.exists():
    italy_boundary = gpd.read_file(italy_shp_path)
    if nodes_gdf.crs != italy_boundary.crs:
        italy_boundary = italy_boundary.to_crs(nodes_gdf.crs)

    italy_boundary.plot(ax=ax, color='#eeeeee', edgecolor='#bcbcbc',
                        linewidth=1, zorder=1)
    print("✅ Base map loaded")

# Set view to Northern Italy with wider horizontal bounds
ax.set_xlim(7.2, 14.0)
ax.set_ylim(43.9, 46.5)

# 4. BIDIRECTIONAL FLOW DETECTION & PLOTTING
max_flow = max([conn['total_flow'] for conn in active_connections]) if active_connections else 1
plotted_types = set()
nodes_list = nodes_gdf['Name'].tolist()

# First pass: identify bidirectional pairs
connection_registry = {}  # {(node1, node2): [conn1, conn2, ...]}

for conn in active_connections:
    conn_name = conn['connection']

    # Match connection string to node names
    from_node, to_node = None, None
    best_match_length = 0

    for n1 in nodes_list:
        if conn_name.startswith(n1):
            remaining = conn_name[len(n1):].strip()
            for n2 in nodes_list:
                if n1 == n2:
                    continue
                if n2 in remaining:
                    match_length = len(n1) + len(n2)
                    if match_length > best_match_length:
                        from_node, to_node = n1, n2
                        best_match_length = match_length

    if from_node and to_node:
        # Store with sorted pair to detect bidirectional routes
        pair = tuple(sorted([from_node, to_node]))
        if pair not in connection_registry:
            connection_registry[pair] = []
        connection_registry[pair].append({
            'from': from_node,
            'to': to_node,
            'data': conn
        })

# Check for bidirectional flows
bidirectional_pairs = {pair for pair, conns in connection_registry.items()
                       if len(conns) > 1}

if bidirectional_pairs:
    print(f"⚠️  WARNING: {len(bidirectional_pairs)} bidirectional flow pairs detected!")
    print("   This may indicate:")
    print("   • Model optimization issue (check mass balance constraints)")
    print("   • Use of existing reversible infrastructure")
    print("   • Temporal flow patterns (seasonal reversal)")

    for pair in list(bidirectional_pairs)[:3]:  # Show first 3 examples
        flows = connection_registry[pair]
        print(f"\n   {pair[0]} ↔ {pair[1]}:")
        for flow in flows:
            print(f"     • {flow['from']} → {flow['to']}: "
                  f"{flow['data']['total_flow']:.0f} t/yr via {flow['data']['network_type']}")

# Second pass: plot with lane offsetting for bidirectional routes
offset_distance = 0.015  # Degrees (adjust based on map scale)

for pair, conns in connection_registry.items():
    is_bidirectional = pair in bidirectional_pairs

    for idx, conn_info in enumerate(conns):
        from_node = conn_info['from']
        to_node = conn_info['to']
        conn = conn_info['data']

        p1 = np.array(coord_dict[from_node])
        p2 = np.array(coord_dict[to_node])
        net_type = conn['network_type']

        # Calculate direction and perpendicular offset
        direction = p2 - p1
        length = np.linalg.norm(direction)

        if length == 0:
            continue

        # Apply offset for bidirectional routes
        if is_bidirectional:
            perp = np.array([-direction[1], direction[0]]) / length
            # Offset in opposite directions for each flow
            offset = perp * offset_distance * (1 if idx == 0 else -1)
            p1_plot = p1 + offset
            p2_plot = p2 + offset
        else:
            p1_plot, p2_plot = p1, p2

        # Flow-based styling
        flow_ratio = conn['total_flow'] / max_flow
        lw = 2 + 8 * flow_ratio

        line_style = '-' if 'Pipeline' in net_type else '--' if 'Railway' in net_type else ':'
        label = net_type if net_type not in plotted_types else ""

        # Plot the connection line
        ax.plot([p1_plot[0], p2_plot[0]], [p1_plot[1], p2_plot[1]],
                color=network_colors.get(net_type, 'gray'),
                linewidth=lw, linestyle=line_style, alpha=0.6,
                zorder=3, label=label)

        # ==========================================
        # ENHANCED ARROW VISIBILITY
        # ==========================================

        # Position arrow at 65% of the route
        arrow_pos = 0.55
        arrow_base = p1_plot + arrow_pos * (p2_plot - p1_plot)
        arrow_tip = p1_plot + (arrow_pos + 0.05) * (p2_plot - p1_plot)

        # Minimum arrow size for visibility
        base_arrow_width = max(2.5, lw * 0.8)
        mutation_scale = max(20, 15 * (lw / 10))

        # White outline for shallow flows
        if flow_ratio < 0.3:
            ax.annotate('',
                        xy=arrow_tip,
                        xytext=arrow_base,
                        arrowprops=dict(
                            arrowstyle='-|>',
                            lw=base_arrow_width + 1.5,
                            color='white',
                            alpha=0.8,
                            mutation_scale=mutation_scale + 2,
                            shrinkA=0,
                            shrinkB=0
                        ),
                        zorder=3.5)

        # Main arrow
        ax.annotate('',
                    xy=arrow_tip,
                    xytext=arrow_base,
                    arrowprops=dict(
                        arrowstyle='-|>',
                        lw=base_arrow_width,
                        color=network_colors.get(net_type, 'gray'),
                        alpha=0.95,
                        mutation_scale=mutation_scale,
                        shrinkA=0,
                        shrinkB=0
                    ),
                    zorder=4)

        plotted_types.add(net_type)

print(f"📊 Plotted {sum(len(c) for c in connection_registry.values())} "
      f"connections with enhanced flow directions")

# 5. Plot Nodes
for _, row in nodes_gdf.iterrows():
    cat = row['category']
    val = row['annual_captured']

    # Determine marker size
    if val > 1_000_000:
        size = 900
    elif val > 500_000:
        size = 600
    elif val > 100_000:
        size = 300
    elif cat == 'storage':
        size = 500
    else:
        size = 100

    color = node_colors.get(cat, node_colors['unknown'])

    if cat == 'storage':
        ax.scatter(row.geometry.x, row.geometry.y, s=size, marker='s',
                   facecolor=color, edgecolor='black', linewidth=1.5, zorder=5)
    else:
        ax.scatter(row.geometry.x, row.geometry.y, s=size, marker='o',
                   facecolor=color, edgecolor='black', linewidth=1.2, zorder=5)

# 6. Legends and Styling
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=node_colors['emitter_full'], edgecolor='k',
          label='Full Capture (~90%)'),
    Patch(facecolor=node_colors['emitter_partial'], edgecolor='k',
          label='Partial Capture'),
    Patch(facecolor=node_colors['emitter_none'], edgecolor='k',
          label='No Capture'),
    Patch(facecolor=node_colors['hub_active'], edgecolor='k',
          label='Active Hub'),
    Patch(facecolor=node_colors['storage'], edgecolor='k',
          label='Storage Site')
]
leg1 = ax.legend(handles=legend_elements, loc='lower right',
                 title="Infrastructure Status", frameon=True, fontsize=11)
ax.add_artist(leg1)

# Transport Legend
if plotted_types:
    leg2 = ax.legend(loc='upper right', title="Transport Mode", fontsize=11)
    ax.add_artist(leg2)

ax.set_title("Northern Italy CO2 Transport Network\n"
             "objective = emissions_minC & MAX storage capacity",
             fontsize=16, pad=20, weight='bold')
ax.set_xlabel("Longitude (°E)", fontsize=12)
ax.set_ylabel("Latitude (°N)", fontsize=12)
ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('co2_network_final_map.png', dpi=300, bbox_inches='tight')
print(f"🎨 Enhanced map saved as: co2_network_final_map.png")
plt.show()

# ============================================================================
# Summary Statistics
# ============================================================================
print(f"\n{'=' * 60}")
print("✅ VISUALIZATION COMPLETE")
print(f"{'=' * 60}")
print(f"📊 Network Summary:")
print(f"   Total Nodes: {len(nodes_gdf)}")
print(f"   Active Connections: {sum(len(c) for c in connection_registry.values())}")
print(f"   Bidirectional Routes: {len(bidirectional_pairs)}")
print(f"\n   Node Categories:")
print(f"     • Full Capture Sites: {sum(nodes_gdf['category'] == 'emitter_full')}")
print(f"     • Partial Capture Sites: {sum(nodes_gdf['category'] == 'emitter_partial')}")
print(f"     • No Capture Sites: {sum(nodes_gdf['category'] == 'emitter_none')}")
print(f"     • Storage Sites: {sum(nodes_gdf['category'] == 'storage')}")
print(f"     • Active Hubs: {sum(nodes_gdf['category'] == 'hub_active')}")
print(f"     • Inactive Hubs: {sum(nodes_gdf['category'] == 'hub_inactive')}")

# Mass balance verification
total_captured = nodes_gdf['annual_captured'].sum()
total_transported = sum(conn['data']['total_flow'] for conns in connection_registry.values()
                        for conn in conns)
print(f"\n   Mass Balance Check:")
print(f"     • Total CO2 Captured: {total_captured:,.0f} t/yr")
print(f"     • Total CO2 Transported: {total_transported:,.0f} t/yr")
if total_captured > 0:
    balance_ratio = (total_transported / total_captured) * 100
    print(f"     • Transport/Capture Ratio: {balance_ratio:.1f}%")
    if balance_ratio > 110:
        print(f"     ⚠️  WARNING: Transport exceeds capture by {balance_ratio - 100:.1f}%")
