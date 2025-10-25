#!/usr/bin/env python3
# Enhanced CCS Network Analysis with Mass Balance Verification and Italy Background
# Reads node coordinates & IDs from Excel: ...\southern_europe\northern_italy_data\geographical_feature\node_metrics.xlsx
# Reads H5 from: ...\southern_europe\resultsImpurities\optimization_results.h5

import h5py
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from collections import defaultdict
from matplotlib.patches import Patch
from shapely.geometry import box
from matplotlib import patheffects as pe  # halo for labels

# ========== GLOBAL FONT CONTROL ==========
# Change BASE_FONT to scale everything (titles, axes, legend, node numbers).
BASE_FONT = 20
TITLE_MAIN_SIZE = BASE_FONT + 4
AXIS_LABEL_SIZE = BASE_FONT
TICK_LABEL_SIZE = BASE_FONT - 2
LEGEND_SIZE = BASE_FONT - 2
NODE_ID_SIZE = BASE_FONT

plt.rcParams.update({
    "axes.titlesize": TITLE_MAIN_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE
})
# ========================================

# Import cmcrameri for navia colormap
try:
    import cmcrameri.cm as cmc
    navia_available = True
    print("CMC colormaps loaded successfully!")
except ImportError:
    print("Warning: cmcrameri not available. Install with: pip install cmcrameri")
    print("Falling back to matplotlib's viridis colormap")
    navia_available = False

warnings.filterwarnings('ignore')


# ------------------- H5 helpers -------------------
def extract_active_flows(h5_file_path, flow_threshold=1e-6):
    """Extract active flow connections from HDF5 file with detailed logging"""
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
                            'total_flow': float(total_flow),
                            'max_flow': float(np.max(flow_data)),
                            'avg_flow': float(np.mean(flow_data))
                        })
                        type_active += 1
                        print(f"    ACTIVE: {connection} (flow: {total_flow:.2e})")
            print(f"  {network_type}: {type_active}/{len(connections)} active")
    return active_connections


def parse_connection_name(connection_name, known_nodes):
    """Parse connection name robustly (from_node at start, to_node at end)"""
    connection_name = str(connection_name).strip()
    nodes_sorted = sorted(known_nodes, key=len, reverse=True)
    from_node = None
    to_node = None
    remaining = ""
    for node in nodes_sorted:
        if connection_name.startswith(node):
            from_node = node
            remaining = connection_name[len(node):].strip()
            break
    if from_node and remaining:
        for node in nodes_sorted:
            if remaining == node or remaining.endswith(node):
                to_node = node
                break
    return from_node, to_node


def identify_node_types(active_connections, all_nodes):
    """Classify nodes as sources, intermediate hubs, or storage sites"""
    node_analysis = defaultdict(lambda: {
        'incoming_flows': [], 'outgoing_flows': [],
        'total_in': 0.0, 'total_out': 0.0, 'node_type': 'unknown'
    })
    for conn in active_connections:
        from_node, to_node = parse_connection_name(conn['connection'], all_nodes)
        if from_node and to_node:
            flow = conn['total_flow']
            node_analysis[from_node]['outgoing_flows'].append({'to': to_node, 'flow': flow})
            node_analysis[from_node]['total_out'] += flow
            node_analysis[to_node]['incoming_flows'].append({'from': from_node, 'flow': flow})
            node_analysis[to_node]['total_in'] += flow
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
    """Print mass-balance summary and return aggregates"""
    print("\n🔍 MASS BALANCE ANALYSIS")
    print("=" * 50)
    sources, intermediates, storage = [], [], []
    for node, data in node_analysis.items():
        if data['node_type'] == 'source':
            sources.append((node, data))
        elif data['node_type'] == 'intermediate':
            intermediates.append((node, data))
        elif data['node_type'] == 'storage':
            storage.append((node, data))
    print("📊 NODE CLASSIFICATION:")
    print(f"  Sources: {len(sources)}")
    print(f"  Intermediates: {len(intermediates)}")
    print(f"  Storage: {len(storage)}")
    total_source_flow = sum(d['total_out'] for _, d in sources)
    total_storage_flow = sum(d['total_in'] for _, d in storage)
    print("\n💧 FLOW TOTALS:")
    print(f"  Total CO2 from sources: {total_source_flow:.2e}")
    print(f"  Total CO2 to storage: {total_storage_flow:.2e}")
    print(f"  Balance difference: {abs(total_source_flow - total_storage_flow):.2e}")
    return {
        'sources': sources, 'intermediates': intermediates, 'storage': storage,
        'total_source_flow': total_source_flow, 'total_storage_flow': total_storage_flow
    }


def analyze_flow_paths(node_analysis, sources, storage):
    """Trace flows from sources to storage (list of paths)"""
    print("\n🛤️  FLOW PATH ANALYSIS")
    print("=" * 50)

    def trace_path(start, visited=None, path=None):
        visited = set() if visited is None else visited
        path = [] if path is None else path
        if start in visited:
            return []
        visited.add(start)
        path.append(start)
        if start in [s[0] for s in storage]:
            return [path.copy()]
        all_paths = []
        for outflow in node_analysis.get(start, {}).get('outgoing_flows', []):
            all_paths += trace_path(outflow['to'], visited.copy(), path.copy())
        return all_paths

    all_paths = []
    for src, sdata in sources:
        for p in trace_path(src):
            all_paths.append({'source': src, 'path': p, 'source_flow': sdata['total_out']})
    for i, fp in enumerate(all_paths, 1):
        print(f"  {i}. {' → '.join(fp['path'])} (src flow {fp['source_flow']:.2e})")
    return all_paths


def explain_flow_mismatch(active_connections, mass_balance_results):
    """Explain why segment-sum != storage total"""
    print("\n🔍 FLOW MISMATCH EXPLANATION")
    print("=" * 50)
    total_pipeline = sum(c['total_flow'] for c in active_connections)
    total_storage = mass_balance_results['total_storage_flow']
    print(f"  Sum of ALL pipeline segment flows: {total_pipeline:.2e}")
    print(f"  Total flow to storage: {total_storage:.2e}")
    print("  The same CO₂ travels across multiple segments, so segment-sum > stored.")


# ------------------- Map helpers -------------------
def load_italy_boundary(path_files_gis):
    """Load Italy boundary shapefile"""
    try:
        shp = path_files_gis / "italy_WGS1984.shp"
        if shp.exists():
            print("✅ Italy boundary loaded")
            return gpd.read_file(shp)
        print(f"⚠️  Italy shapefile not found: {shp}")
        return None
    except Exception as e:
        print(f"⚠️  Could not load Italy boundary: {e}")
        return None


def get_italy_northern_region(nodes_gdf):
    """Bounding box around nodes with small buffer"""
    minx, miny, maxx, maxy = nodes_gdf.total_bounds
    buf = 0.5
    return box(minx - buf, miny - buf, maxx + buf, maxy + buf)


# ------------------- Label helper -------------------
def label_nodes(ax, nodes_gdf, fontsize=NODE_ID_SIZE,
                dy_up=0.085, default_dx=0.0,
                special_offsets=None):
    """
    Draw node numbers using the Excel 'ID' values (no renumbering).
    Offsets via special_offsets={id:(dx, dy)} override the default (0, dy_up).
    """
    if special_offsets is None:
        special_offsets = {}
    for _, row in nodes_gdf.iterrows():
        node_id = int(row["ID"])
        x, y = row.geometry.x, row.geometry.y
        dx, dy = special_offsets.get(node_id, (default_dx, dy_up))
        ax.text(
            x + dx, y + dy, str(node_id),
            ha="center", va="center", fontsize=fontsize, color="black", zorder=30,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )


# ------------------- Plot routine -------------------
def create_enhanced_network_plot_with_italy(nodes_gdf, active_connections, coord_dict, node_analysis, path_files_gis):
    """Plot network over Northern Italy background and add node numbers like the other plot"""
    if navia_available:
        navia_cmap = cmc.navia
        network_colors = {'CO2_Pipeline': navia_cmap(0.15),
                          'CO2Railway': navia_cmap(0.85),
                          'CO2Truck': navia_cmap(0.5)}
    else:
        viridis = plt.cm.viridis
        network_colors = {'CO2_Pipeline': viridis(0.15),
                          'CO2Railway': viridis(0.85),
                          'CO2Truck': viridis(0.5)}

    italy = load_italy_boundary(path_files_gis)
    fig, ax = plt.subplots(figsize=(20, 16))

    if italy is not None and nodes_gdf.crs != italy.crs:
        italy = italy.to_crs(nodes_gdf.crs)

    if italy is not None:
        north_box = gpd.GeoDataFrame(geometry=[get_italy_northern_region(nodes_gdf)], crs=italy.crs)
        italy_n = gpd.clip(italy, north_box)
        italy_n.plot(ax=ax, color='lightgray', alpha=0.4, edgecolor='black', linewidth=1.2, zorder=1)
        italy_n.boundary.plot(ax=ax, color='black', linewidth=1.5, zorder=2)
        ax.set_xlim(8.5, 12.7)
        ax.set_ylim(44.3, 45.8)
    else:
        ax.set_xlim(8.5, 12.7)
        ax.set_ylim(44.3, 45.8)

    node_type_colors = {'source': '#000000', 'storage': '#43A047',
                        'intermediate': '#888888', 'inactive': '#CCCCCC'}

    # Size categories (mocked, cosmetic)
    def node_size(node, kind):
        if kind == 'source':
            rng = (200, 1000)
            return np.linspace(*rng, 6, dtype=int)[hash(node) % 6]
        return 150 if kind in ('storage', 'intermediate') else 100

    # Draw nodes (patches)
    xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
    yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
    scale = min(xspan, yspan) / 1200
    from matplotlib.patches import Circle, Rectangle

    for _, row in nodes_gdf.iterrows():
        name = row['Name']
        x, y = row.geometry.x, row.geometry.y
        kind = node_analysis.get(name, {}).get('node_type', 'inactive')
        color = node_type_colors.get(kind, '#CCCCCC')
        r = np.sqrt(node_size(name, kind)) * scale
        if kind == 'storage':
            s = r * 1.8
            ax.add_patch(Rectangle((x - s/2, y - s/2), s, s, facecolor=color,
                                   edgecolor='black', linewidth=3, zorder=6))
        else:
            ax.add_patch(Circle((x, y), r, facecolor=color,
                                edgecolor='black', linewidth=3, zorder=6))

    # Label IDs from Excel (global up-shift; custom tweaks optional)
    label_nodes(
        ax, nodes_gdf,
        fontsize=NODE_ID_SIZE,
        dy_up=0.085,
        special_offsets={
            # tweak if needed; examples:
            14: (-0.05, 0.085),
            10: (0.0, -0.10),
        },
    )

    # Draw connections (inlet full opacity, rest 35%)
    styles = {
        'CO2_Pipeline': {'color': network_colors['CO2_Pipeline'], 'ls': '-',  'alpha': 0.35, 'label': 'CO2 Pipeline'},
        'CO2Railway':   {'color': network_colors['CO2Railway'],   'ls': '--', 'alpha': 0.35, 'label': 'CO2 Railway'},
        'CO2Truck':     {'color': network_colors['CO2Truck'],     'ls': ':',  'alpha': 0.35, 'label': 'CO2 Truck'}
    }
    max_flow = max((c['total_flow'] for c in active_connections), default=1.0)
    shown = set()
    for c in active_connections:
        f, t = parse_connection_name(c['connection'], coord_dict.keys())
        if not (f and t and f in coord_dict and t in coord_dict):
            continue
        fx, fy = coord_dict[f]
        tx, ty = coord_dict[t]
        st = styles.get(c['network_type'], {'color': 'gray', 'ls': '-', 'alpha': 0.35, 'label': 'Other'})
        lw = 6 + 10 * (c['total_flow'] / max_flow)
        dx, dy = tx - fx, ty - fy
        L = np.hypot(dx, dy) or 1e-9
        L_in = 0.08  # fixed inlet length in degrees
        frac = min(L_in / L, 0.4)
        ix, iy = fx + dx * frac, fy + dy * frac
        ax.plot([fx, ix], [fy, iy], color=st['color'], lw=lw, ls=st['ls'], alpha=1.0, zorder=4)
        ax.plot([ix, tx], [iy, ty], color=st['color'], lw=lw, ls=st['ls'],
                alpha=st['alpha'], zorder=3,
                label=st['label'] if c['network_type'] not in shown else "")
        shown.add(c['network_type'])

    # Legends
    if shown:
        ax.legend(title='Transport Mode', loc='upper right', framealpha=0.95,
                  fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE)

    node_legend = [
        Patch(facecolor='#000000', label='CO2 Sources'),
        Patch(facecolor='#888888', label='Intermediate Hubs'),
        Patch(facecolor='#43A047', label='Storage Sites'),
        Patch(facecolor='#CCCCCC', label='Inactive Nodes')
    ]
    if italy is not None:
        node_legend.append(Patch(facecolor='lightgray', alpha=0.4, edgecolor='black', label='Italy Boundary'))
    ax.legend(
        handles=node_legend,
        title='Node Type',
        loc='lower left',
        bbox_to_anchor=(0.02, 0.02),
        framealpha=0.95,
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
    )

    # Cosmetics
    ax.set_xlabel('Longitude (°E)', fontsize=AXIS_LABEL_SIZE, weight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=AXIS_LABEL_SIZE, weight='bold')
    ax.set_title('CO2 Transport Network in Northern Italy\n'
                 'Optimized CCS Infrastructure with Mass Balance Verification',
    fontsize=TITLE_MAIN_SIZE, weight='bold', pad=25)
    ax.grid(True, alpha=0.3, ls='--', lw=0.5)
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f9fa')
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    plt.savefig('enhanced_co2_network_italy.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    return len(active_connections)


# ------------------- Main -------------------
def main():
    """Main analysis function with enhanced mass balance verification and Italy background"""
    print("Enhanced CO2 Transport Network Analysis with Italy Background")
    print("=" * 60)

    # ----------- PATHS -----------
    path_root = Path(r"C:\Users\mockp\PycharmProjects\AdOpT-NET0_XiaoY\southern_europe")
    path_files_gis = path_root / "northern_italy_data" / "raw_data" / "gis_data"
    # Excel with authoritative node locations/IDs
    path_files_node_flux = path_root / "northern_italy_data" / "geographical_feature"
    node_xlsx = path_files_node_flux / "node_metrics.xlsx"

    # H5 results
    results_data_path = path_root / "resultsImpurities"
    h5_file_path = results_data_path / "optimization_results.h5"
    # -----------------------------

    # Checks
    if not h5_file_path.exists():
        print(f"❌ H5 file not found: {h5_file_path}")
        return
    if not node_xlsx.exists():
        print(f"❌ Excel not found: {node_xlsx}")
        return

    # --- Load nodes from Excel (authoritative for ID + coordinates) ---
    print("📂 Loading node positions from Excel...")
    nodes_df = pd.read_excel(node_xlsx, sheet_name="nodes")  # change if your sheet has a different name

    # Flexible column detection
    def pick(colnames, *cands):
        cands_l = [c.lower() for c in cands]
        for c in colnames:
            if c.strip().lower() in cands_l:
                return c
        raise KeyError(f"Missing expected columns {cands} in {list(colnames)}")

    col_id  = pick(nodes_df.columns, "ID", "Node_ID", "node_id")
    col_name= pick(nodes_df.columns, "Name", "Node", "node", "node_name")
    col_lon = pick(nodes_df.columns, "Lon", "Longitude", "X", "lon", "longitude")
    col_lat = pick(nodes_df.columns, "Lat", "Latitude", "Y", "lat", "latitude")

    nodes_df = (nodes_df[[col_id, col_name, col_lon, col_lat]]
                .rename(columns={col_id: "ID", col_name: "Name", col_lon: "Lon", col_lat: "Lat"}))

    # Keep only valid rows; coerce ID to int; drop duplicate IDs
    nodes_df = nodes_df.dropna(subset=["ID", "Lon", "Lat"]).copy()
    nodes_df["ID"] = nodes_df["ID"].astype(int)
    nodes_df = nodes_df.drop_duplicates(subset="ID", keep="first")

    nodes_gdf = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df["Lon"], nodes_df["Lat"]),
        crs="EPSG:4326",
    ).sort_values("ID").reset_index(drop=True)

    print(f"✅ Loaded {len(nodes_gdf)} nodes from Excel; IDs: {sorted(nodes_gdf['ID'].tolist())}")
    print(f"📍 Node extent: {nodes_gdf.total_bounds}")

    # Build coordinate dictionary by Name for edge drawing
    coord_dict = {row['Name']: (row.geometry.x, row.geometry.y) for _, row in nodes_gdf.iterrows()}

    print("\n📊 Extracting flow data...")
    active_connections = extract_active_flows(h5_file_path)
    if not active_connections:
        print("❌ No active connections found!")
        return

    print("\n🔬 Performing network analysis...")
    node_analysis = identify_node_types(active_connections, coord_dict.keys())
    mass_balance = verify_mass_balance(node_analysis)
    analyze_flow_paths(node_analysis, mass_balance['sources'], mass_balance['storage'])
    explain_flow_mismatch(active_connections, mass_balance)

    print("\n🎨 Creating enhanced visualization with Italy boundary...")
    plotted_count = create_enhanced_network_plot_with_italy(
        nodes_gdf, active_connections, coord_dict, node_analysis, path_files_gis
    )

    print("\n✅ ANALYSIS COMPLETE")
    print("📊 Network Summary:")
    print(f"   • {len(mass_balance['sources'])} CO2 sources")
    print(f"   • {len(mass_balance['intermediates'])} intermediate hubs")
    print(f"   • {len(mass_balance['storage'])} storage sites")
    print(f"   • {plotted_count} active transport connections")
    print("📁 Saved: enhanced_co2_network_italy.png")


if __name__ == "__main__":
    main()
