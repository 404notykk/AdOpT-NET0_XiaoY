import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from matplotlib import patheffects as pe
import matplotlib.gridspec as gridspec

# ======= GLOBAL FONT CONTROL =======
BASE_FONT = 20                 # change this to scale everything
TITLE_MAIN_SIZE = BASE_FONT + 4
SUBPLOT_TITLE_SIZE = BASE_FONT + 2
AXIS_LABEL_SIZE = BASE_FONT
TICK_LABEL_SIZE = BASE_FONT - 2
LEGEND_SIZE = BASE_FONT - 2
NODE_ID_SIZE = BASE_FONT

plt.rcParams.update({
    "axes.titlesize": SUBPLOT_TITLE_SIZE,   # default for axes titles
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE
})
# ===================================

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

# ---------- PATHS ----------
path_data_case_study = Path("../northern_italy_data")
path_files_gis = path_data_case_study / "raw_data/gis_data"
path_files_node_flux = path_data_case_study / "geographical_feature"
node_xlsx = path_files_node_flux / "node_metrics.xlsx"
# ---------------------------

# Load boundary and routes (as before)
italy = gpd.read_file(path_files_gis / "italy_WGS1984.shp")
routes_pipeline = gpd.read_file(path_files_gis / "routes_distances_pipeline.shp")
routes_railway = gpd.read_file(path_files_gis / "routes_distances_railway.shp")
routes_truck = gpd.read_file(path_files_gis / "routes_distances_truck.shp")

# Load network matrices for transport directions
network_pipeline = pd.read_excel(node_xlsx, index_col=0, sheet_name='pipeline')
network_truck = pd.read_excel(node_xlsx, index_col=0, sheet_name='truck')
network_railway = pd.read_excel(node_xlsx, index_col=0, sheet_name='railway')

# --- NEW: Load node coordinates & IDs from Excel (authoritative) ---
def pick(colnames, *cands):
    cands_l = [c.lower() for c in cands]
    for c in colnames:
        if c.strip().lower() in cands_l:
            return c
    raise KeyError(f"Missing expected columns {cands} in {list(colnames)}")

# Try to read a sheet that lists nodes/coordinates. Adjust if your sheet is named differently.
nodes_df = pd.read_excel(node_xlsx, sheet_name="nodes")

col_id = pick(nodes_df.columns, "ID", "Node_ID", "node_id")
col_name = pick(nodes_df.columns, "Name", "Node", "node", "node_name")
col_lon = pick(nodes_df.columns, "Lon", "Longitude", "X", "lon", "longitude")
col_lat = pick(nodes_df.columns, "Lat", "Latitude", "Y", "lat", "latitude")

nodes_df = nodes_df[[col_id, col_name, col_lon, col_lat]].rename(
    columns={"ID": "ID", col_id: "ID", col_name: "Name", col_lon: "Lon", col_lat: "Lat"}
)

nodes_selected = gpd.GeoDataFrame(
    nodes_df,
    geometry=gpd.points_from_xy(nodes_df["Lon"], nodes_df["Lat"]),
    crs="EPSG:4326",
).sort_values("ID").reset_index(drop=True)
# -------------------------------------------------------------------

print("Data loaded successfully!")
print(f"Italy boundary: {italy.shape[0]} features")
print(f"Selected nodes (from Excel): {nodes_selected.shape[0]} nodes")
print(f"Pipeline routes: {routes_pipeline.shape[0]} routes")
print(f"Railway routes: {routes_railway.shape[0]} routes")
print(f"Truck routes: {routes_truck.shape[0]} routes")

# Colors
def setup_navia_colors():
    if navia_available:
        navia_cmap = cmc.navia
        route_colors = {'pipeline': navia_cmap(0.15), 'truck': navia_cmap(0.5), 'railway': navia_cmap(0.85)}
        colormap = navia_cmap
    else:
        viridis_cmap = plt.cm.viridis
        route_colors = {'pipeline': viridis_cmap(0.15), 'truck': viridis_cmap(0.5), 'railway': viridis_cmap(0.85)}
        colormap = viridis_cmap
    return route_colors, colormap

route_colors, colormap = setup_navia_colors()

# --- ID-aware labeling (uses Excel 'ID' rather than enumerate) ---
def _node_ids_in_plot_order(gdf):
    if "ID" in gdf.columns:
        try:
            ids = gdf["ID"].astype(int).tolist()
            # Keep the order of rows; we already sorted by ID
            return ids
        except Exception:
            pass
    # Fallback to 1..N
    return list(range(1, len(gdf) + 1))

def label_nodes(ax, gdf, fontsize=NODE_ID_SIZE, dy_up=0.035, default_dx=0.0, special_offsets=None):
    """Write node IDs with halo. special_offsets={node_id:(dx,dy)}"""
    if special_offsets is None:
        special_offsets = {}
    ids = _node_ids_in_plot_order(gdf)
    for (_, row), node_id in zip(gdf.iterrows(), ids):
        x, y = row.geometry.x, row.geometry.y
        dx, dy = special_offsets.get(int(node_id), (default_dx, dy_up))
        ax.text(
            x + dx, y + dy, str(int(node_id)),
            ha="center", va="center", fontsize=fontsize, color="black", zorder=30,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )

def get_route_directionality_fixed(routes_gdf, network_matrix, route_type):
    route_directions = {}
    debug_info = []
    print(f"\nAnalyzing {route_type} routes with FIXED logic...")
    print(f"Network matrix shape: {network_matrix.shape}")
    for idx, route in routes_gdf.iterrows():
        try:
            from_node_id = None
            to_node_id = None
            method_used = "unknown"
            if 'Node' in route.index and pd.notna(route['Node']):
                node_str = str(route['Node']).strip()
                separators = [',', '-', ';', '|', ' ']
                for sep in separators:
                    if sep in node_str:
                        node_parts = node_str.split(sep)
                        if len(node_parts) >= 2:
                            try:
                                from_node_id = int(node_parts[0].strip())
                                to_node_id = int(node_parts[1].strip())
                                method_used = f"node_column_{sep}_separated"
                                break
                            except ValueError:
                                continue
                        break
            line = route.geometry
            start_point = Point(line.coords[0])
            end_point = Point(line.coords[-1])
            start_distances = nodes_selected.geometry.distance(start_point)
            end_distances = nodes_selected.geometry.distance(end_point)
            closest_to_start_idx = start_distances.idxmin()
            closest_to_end_idx = end_distances.idxmin()
            # Because nodes_selected is sorted by ID starting at 1:
            geometry_start_node = int(nodes_selected.iloc[closest_to_start_idx]["ID"])
            geometry_end_node = int(nodes_selected.iloc[closest_to_end_idx]["ID"])
            if from_node_id is None or to_node_id is None:
                from_node_id = geometry_start_node
                to_node_id = geometry_end_node
                method_used = "geometry_fallback"
            forward_connection = backward_connection = False
            forward_value = backward_value = 0
            if (from_node_id in network_matrix.index) and (to_node_id in network_matrix.columns):
                forward_value = network_matrix.loc[from_node_id, to_node_id]
                forward_connection = forward_value > 0
            if (to_node_id in network_matrix.index) and (from_node_id in network_matrix.columns):
                backward_value = network_matrix.loc[to_node_id, from_node_id]
                backward_connection = backward_value > 0
            if forward_connection and backward_connection:
                direction = 'bidirectional'; inlet_position = 'both_ends'; flow_origin_node = None
            elif forward_connection:
                direction = 'forward'; flow_origin_node = from_node_id
                inlet_position = 'start' if geometry_start_node == from_node_id else 'end'
            elif backward_connection:
                direction = 'backward'; flow_origin_node = to_node_id
                inlet_position = 'start' if geometry_start_node == to_node_id else 'end'
            else:
                direction = 'none'; inlet_position = 'start'; flow_origin_node = None
            route_directions[idx] = {
                'direction': direction, 'inlet_position': inlet_position,
                'from_node': from_node_id, 'to_node': to_node_id,
                'geometry_start_node': geometry_start_node, 'geometry_end_node': geometry_end_node,
                'flow_origin_node': flow_origin_node, 'method': method_used,
                'forward_value': forward_value, 'backward_value': backward_value
            }
            debug_info.append({'direction': direction, 'method': method_used})
        except Exception as e:
            route_directions[idx] = {'direction':'error','inlet_position':'start','from_node':None,'to_node':None,
                                     'geometry_start_node':None,'geometry_end_node':None,'flow_origin_node':None,
                                     'method':'error','forward_value':0,'backward_value':0}
    direction_counts = {}
    method_counts = {}
    for info in debug_info:
        direction_counts[info['direction']] = direction_counts.get(info['direction'], 0) + 1
        method_counts[info['method']] = method_counts.get(info['method'], 0) + 1
    print(f"{route_type} direction analysis:")
    for k,v in direction_counts.items(): print(f"  {k}: {v} routes")
    print(f"{route_type} method analysis:")
    for k,v in method_counts.items(): print(f"  {k}: {v} routes")
    return route_directions

def plot_route_with_inlet_emphasis_fixed(ax, coords, color, direction_info, linewidth, base_alpha):
    direction = direction_info.get('direction', 'unknown')
    inlet_position = direction_info.get('inlet_position', 'start')
    static_counter = getattr(plot_route_with_inlet_emphasis_fixed, 'counter', 0)
    plot_route_with_inlet_emphasis_fixed.counter = static_counter + 1
    rest_alpha = 0.35
    if direction == 'bidirectional':
        if len(coords) == 2:
            start_point, end_point = coords[0], coords[1]
            total_distance = ((end_point[0]-start_point[0])**2 + (end_point[1]-start_point[1])**2)**0.5
            target_inlet_distance = 0.15
            if total_distance > 2*target_inlet_distance:
                inlet_ratio = target_inlet_distance / total_distance
                sx = start_point[0] + inlet_ratio*(end_point[0]-start_point[0])
                sy = start_point[1] + inlet_ratio*(end_point[1]-start_point[1])
                ex = start_point[0] + (1-inlet_ratio)*(end_point[0]-start_point[0])
                ey = start_point[1] + (1-inlet_ratio)*(end_point[1]-start_point[1])
                start_split = (sx, sy); end_split = (ex, ey)
                gpd.GeoSeries([LineString([start_split, end_split])]).plot(ax=ax, color=color, linewidth=linewidth, alpha=0.4, zorder=6)
                gpd.GeoSeries([LineString([start_point, start_split])]).plot(ax=ax, color=color, linewidth=linewidth*1.2, alpha=0.6, zorder=7)
                gpd.GeoSeries([LineString([end_split, end_point])]).plot(ax=ax, color=color, linewidth=linewidth*1.2, alpha=0.6, zorder=7)
            else:
                gpd.GeoSeries([LineString([start_point, end_point])]).plot(ax=ax, color=color, linewidth=linewidth*1.1, alpha=0.55, zorder=6)
        else:
            gpd.GeoSeries([LineString(coords)]).plot(ax=ax, color=color, linewidth=linewidth*1.1, alpha=0.55, zorder=6)
        return
    inlet_at_start = (inlet_position == 'start')
    if len(coords) == 2:
        start_point, end_point = coords[0], coords[1]
        total_distance = ((end_point[0]-start_point[0])**2 + (end_point[1]-start_point[1])**2)**0.5
        target_inlet_distance = 0.15
        if total_distance > target_inlet_distance:
            split_ratio = target_inlet_distance/total_distance
            if not inlet_at_start: split_ratio = 1.0 - split_ratio
        else:
            split_ratio = 1.0 if inlet_at_start else 0.0
        if 0.05 < split_ratio < 0.95:
            sx = start_point[0] + split_ratio*(end_point[0]-start_point[0])
            sy = start_point[1] + split_ratio*(end_point[1]-start_point[1])
            split_point = (sx, sy)
            if inlet_at_start:
                gpd.GeoSeries([LineString([split_point, end_point])]).plot(ax=ax, color=color, linewidth=linewidth, alpha=rest_alpha, zorder=7)
                gpd.GeoSeries([LineString([start_point, split_point])]).plot(ax=ax, color=color, linewidth=linewidth*1.5, alpha=1.0, zorder=9)
            else:
                gpd.GeoSeries([LineString([start_point, split_point])]).plot(ax=ax, color=color, linewidth=linewidth, alpha=rest_alpha, zorder=7)
                gpd.GeoSeries([LineString([split_point, end_point])]).plot(ax=ax, color=color, linewidth=linewidth*1.5, alpha=1.0, zorder=9)
        else:
            gpd.GeoSeries([LineString([start_point, end_point])]).plot(ax=ax, color=color, linewidth=linewidth*1.5, alpha=1.0, zorder=9)
        return
    target_inlet_distance = 0.15
    cumulative_distances = [0.0]; total_distance = 0.0
    for i in range(1, len(coords)):
        seg = ((coords[i][0]-coords[i-1][0])**2 + (coords[i][1]-coords[i-1][1])**2)**0.5
        total_distance += seg; cumulative_distances.append(total_distance)
    if total_distance > target_inlet_distance:
        split_ratio = target_inlet_distance/total_distance
        if not inlet_at_start: split_ratio = 1.0 - split_ratio
    else:
        split_ratio = 1.0 if inlet_at_start else 0.0
    if 0.05 < split_ratio < 0.95:
        target_split = split_ratio * total_distance
        split_idx = len(coords)-1
        for i, cum in enumerate(cumulative_distances):
            if cum >= target_split:
                split_idx = max(1, i)
                break
        split_idx = min(split_idx, len(coords)-1)
        if inlet_at_start:
            if split_idx < len(coords)-1:
                gpd.GeoSeries([LineString(coords[split_idx-1:])]).plot(ax=ax, color=color, linewidth=linewidth, alpha=0.35, zorder=7)
            gpd.GeoSeries([LineString(coords[:split_idx+1])]).plot(ax=ax, color=color, linewidth=linewidth*1.5, alpha=1.0, zorder=9)
        else:
            if split_idx > 0:
                gpd.GeoSeries([LineString(coords[:split_idx+1])]).plot(ax=ax, color=color, linewidth=linewidth, alpha=0.35, zorder=7)
            gpd.GeoSeries([LineString(coords[split_idx:])]).plot(ax=ax, color=color, linewidth=linewidth*1.5, alpha=1.0, zorder=9)
    else:
        gpd.GeoSeries([LineString(coords)]).plot(ax=ax, color=color, linewidth=linewidth*1.5, alpha=1.0, zorder=9)

def plot_simple_route(ax, route, color, linewidth=2, alpha=0.7):
    line = route.geometry
    gpd.GeoSeries([line]).plot(ax=ax, color=color, linewidth=linewidth, alpha=alpha, zorder=5)

def plot_route_with_enhanced_direction_fixed(ax, route, color, direction_info, linewidth=4, alpha=0.7, show_inlet=True):
    line = route.geometry
    coords = list(line.coords)
    if len(coords) < 2: return
    if show_inlet:
        plot_route_with_inlet_emphasis_fixed(ax, coords, color, direction_info, linewidth, alpha)
    else:
        gpd.GeoSeries([line]).plot(ax=ax, color=color, linewidth=linewidth, alpha=alpha, zorder=5)

print("\nAnalyzing route directions using FIXED logic...")
plot_route_with_inlet_emphasis_fixed.counter = 0
pipeline_directions = get_route_directionality_fixed(routes_pipeline, network_pipeline, 'Pipeline')
truck_directions = get_route_directionality_fixed(routes_truck, network_truck, 'Truck')
railway_directions = get_route_directionality_fixed(routes_railway, network_railway, 'Railway')

north_italy_bounds = {'minx': 8.5, 'maxx': 13.0, 'miny': 44.25, 'maxy': 46.0}

# Create figure with 1 detailed subplot instead of 3
fig = plt.figure(figsize=(22, 12))
gs = gridspec.GridSpec(1, 2, hspace=0.4, wspace=0.25, width_ratios=[1.2, 1])

# Left overview (pipelines only)
ax1 = fig.add_subplot(gs[:, 0])
italy.boundary.plot(ax=ax1, color='black', linewidth=1.5, alpha=0.8)
italy.plot(ax=ax1, color='lightgray', alpha=0.3)
for _, route in routes_pipeline.iterrows():
    plot_simple_route(ax1, route, route_colors['pipeline'], linewidth=2, alpha=1.0)
nodes_selected.plot(ax=ax1, color='red', markersize=120, alpha=1.0,
                    edgecolors='white', linewidth=2.5, zorder=20)

from matplotlib.patches import Rectangle
rect = Rectangle((north_italy_bounds['minx'], north_italy_bounds['miny']),
                 north_italy_bounds['maxx'] - north_italy_bounds['minx'],
                 north_italy_bounds['maxy'] - north_italy_bounds['miny'],
                 linewidth=3, edgecolor='red', facecolor='none', linestyle='--')
ax1.add_patch(rect)

ax1.set_title('Italy Transportation Routes Overview\n(Red box shows detailed area)',
              fontsize=TITLE_MAIN_SIZE, fontweight='bold', pad=20)
ax1.set_xlabel('Longitude', fontsize=AXIS_LABEL_SIZE)
ax1.set_ylabel('Latitude', fontsize=AXIS_LABEL_SIZE)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal')

# ---- Right side: Pipeline network only ----
ax = fig.add_subplot(gs[:, 1])
italy.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.6)
italy.plot(ax=ax, color='lightgray', alpha=0.2)

for idx, route in routes_pipeline.iterrows():
    plot_route_with_enhanced_direction_fixed(ax, route, route_colors['pipeline'], pipeline_directions[idx],
                                             linewidth=4, alpha=1.0, show_inlet=True)

nodes_selected.plot(ax=ax, color='black', markersize=100, alpha=1.0,
                    edgecolors='white', linewidth=2.5, zorder=25)

# Label node IDs from Excel (with the same halo/offset style)
label_nodes(ax, nodes_selected, fontsize=NODE_ID_SIZE, dy_up=0.085,
            special_offsets={14: (-0.05, 0.085), 10: (0.0, -0.1)})

ax.set_xlim(north_italy_bounds['minx'], north_italy_bounds['maxx'])
ax.set_ylim(north_italy_bounds['miny'], north_italy_bounds['maxy'])
ax.set_title(f'Pipeline Network - Northern Italy\n({len(routes_pipeline)} routes)',
             fontsize=SUBPLOT_TITLE_SIZE, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel('Latitude', fontsize=AXIS_LABEL_SIZE)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_aspect('equal')

legend_elements = [
    plt.Line2D([0], [0], color=route_colors['pipeline'], lw=4, label='Pipeline'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10,
               markeredgecolor='white', markeredgewidth=2, label='Network Nodes', linestyle='None'),
    plt.Line2D([0], [0], color='black', lw=6, alpha=1.0, label='Inlet segment (100% opacity)'),
    plt.Line2D([0], [0], color='gray', lw=4, alpha=0.35, label='Route middle/tail (35% opacity)')
]
ax1.legend(handles=legend_elements, loc='lower left', fontsize=LEGEND_SIZE,
           frameon=True, fancybox=True, shadow=True)

output_filename = "italy_transportation_fixed_inlets.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
print(f"\nPlot saved as: {output_filename}")

plt.show()
