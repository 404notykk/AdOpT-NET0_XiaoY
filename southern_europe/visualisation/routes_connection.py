import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

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

# Load data
path_data_case_study = Path("../northern_italy_data")
path_files_gis = path_data_case_study / "raw_data/gis_data"
path_files_node_flux = path_data_case_study / "geographical_feature"

italy = gpd.read_file(path_files_gis / "italy_WGS1984.shp")
nodes_selected = gpd.read_file(path_files_gis / "nodes_italy_14.shp")
routes_pipeline = gpd.read_file(path_files_gis / "routes_distances_pipeline.shp")
routes_railway = gpd.read_file(path_files_gis / "routes_distances_railway.shp")
routes_truck = gpd.read_file(path_files_gis / "routes_distances_truck.shp")

# Load network matrices for transport directions
network_pipeline = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0, sheet_name='pipeline')
network_truck = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0, sheet_name='truck')
network_railway = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0, sheet_name='railway')

# Print basic info about the loaded data
print("Data loaded successfully!")
print(f"Italy boundary: {italy.shape[0]} features")
print(f"Selected nodes: {nodes_selected.shape[0]} nodes")
print(f"Pipeline routes: {routes_pipeline.shape[0]} routes")
print(f"Railway routes: {routes_railway.shape[0]} routes")
print(f"Truck routes: {routes_truck.shape[0]} routes")


# Setup specific colors from navia colormap
def setup_navia_colors():
    """Setup specific colors from CMC navia colormap for each route type"""
    if navia_available:
        navia_cmap = cmc.navia
        route_colors = {
            'pipeline': navia_cmap(0.15),
            'truck': navia_cmap(0.5),
            'railway': navia_cmap(0.85)
        }
        colormap = navia_cmap
    else:
        viridis_cmap = plt.cm.viridis
        route_colors = {
            'pipeline': viridis_cmap(0.15),
            'truck': viridis_cmap(0.5),
            'railway': viridis_cmap(0.85)
        }
        colormap = viridis_cmap
    return route_colors, colormap


# Get colors
route_colors, colormap = setup_navia_colors()


def get_route_directionality_fixed(routes_gdf, network_matrix, route_type):
    """
    FIXED version that properly handles inlet positioning based on flow direction and geometry
    """
    route_directions = {}
    debug_info = []

    print(f"\nAnalyzing {route_type} routes with FIXED logic...")
    print(f"Network matrix shape: {network_matrix.shape}")

    for idx, route in routes_gdf.iterrows():
        try:
            from_node_id = None
            to_node_id = None
            method_used = "unknown"

            # Method 1: Handle different separators in 'Node' column
            if 'Node' in route.index and pd.notna(route['Node']):
                node_str = str(route['Node']).strip()
                print(f"  Processing route {idx} with Node column: '{node_str}'")

                # Try different separators
                separators = [',', '-', ';', '|', ' ']
                for sep in separators:
                    if sep in node_str:
                        node_parts = node_str.split(sep)
                        if len(node_parts) >= 2:
                            try:
                                # Convert directly to integers (matching matrix indices)
                                from_node_id = int(node_parts[0].strip())
                                to_node_id = int(node_parts[1].strip())
                                method_used = f"node_column_{sep}_separated"
                                print(f"    SUCCESS: Parsed {from_node_id} -> {to_node_id}")
                                break
                            except ValueError:
                                print(
                                    f"    Could not convert to integers: {node_parts[0].strip()}, {node_parts[1].strip()}")
                                continue
                        break

            # Method 2: Get geometry endpoints and find nearest nodes
            line = route.geometry
            start_point = Point(line.coords[0])
            end_point = Point(line.coords[-1])

            # Find which actual nodes are closest to geometry endpoints
            start_distances = nodes_selected.geometry.distance(start_point)
            end_distances = nodes_selected.geometry.distance(end_point)

            closest_to_start_idx = start_distances.idxmin()
            closest_to_end_idx = end_distances.idxmin()

            # Convert 0-based geometry indices to 1-based node IDs to match matrix
            geometry_start_node = closest_to_start_idx + 1
            geometry_end_node = closest_to_end_idx + 1

            # Fallback if Node column parsing failed
            if from_node_id is None or to_node_id is None:
                from_node_id = geometry_start_node
                to_node_id = geometry_end_node
                method_used = "geometry_fallback"
                print(f"    FALLBACK: Using geometry-based nodes {from_node_id} -> {to_node_id}")

            print(f"    Node column: {from_node_id}->{to_node_id}")
            print(f"    Geometry: start near node {geometry_start_node}, end near node {geometry_end_node}")

            # Check network matrix for directionality
            forward_connection = False
            backward_connection = False
            forward_value = 0
            backward_value = 0

            if from_node_id is not None and to_node_id is not None:
                try:
                    # Check forward direction
                    if from_node_id in network_matrix.index and to_node_id in network_matrix.columns:
                        forward_value = network_matrix.loc[from_node_id, to_node_id]
                        forward_connection = forward_value > 0
                        print(f"    Forward connection {from_node_id}->{to_node_id}: {forward_value}")

                    # Check backward direction
                    if to_node_id in network_matrix.index and from_node_id in network_matrix.columns:
                        backward_value = network_matrix.loc[to_node_id, from_node_id]
                        backward_connection = backward_value > 0
                        print(f"    Backward connection {to_node_id}->{from_node_id}: {backward_value}")

                except Exception as e:
                    print(f"    Error checking network matrix: {e}")

            # FIXED LOGIC: Determine direction and inlet position based on ACTUAL flow direction
            if forward_connection and backward_connection:
                direction = 'bidirectional'
                inlet_position = 'both_ends'
                flow_origin_node = None  # Both nodes are origins
                print(f"    BIDIRECTIONAL flow -> inlet at BOTH ends")
            elif forward_connection:
                direction = 'forward'
                flow_origin_node = from_node_id  # Flow starts from from_node
                # Determine where the flow origin node is in the geometry
                if geometry_start_node == from_node_id:
                    inlet_position = 'start'  # Origin is at geometry start
                    print(f"    FORWARD flow: origin node {from_node_id} at geometry START -> inlet at START")
                elif geometry_end_node == from_node_id:
                    inlet_position = 'end'  # Origin is at geometry end
                    print(f"    FORWARD flow: origin node {from_node_id} at geometry END -> inlet at END")
                else:
                    # This shouldn't happen if geometry parsing is correct
                    inlet_position = 'start'  # fallback
                    print(
                        f"    FORWARD flow: WARNING - origin node {from_node_id} not clearly at geometry endpoints -> default START")
            elif backward_connection:
                direction = 'backward'
                flow_origin_node = to_node_id  # Flow starts from to_node (backward direction)
                # Determine where the flow origin node is in the geometry
                if geometry_start_node == to_node_id:
                    inlet_position = 'start'  # Origin is at geometry start
                    print(f"    BACKWARD flow: origin node {to_node_id} at geometry START -> inlet at START")
                elif geometry_end_node == to_node_id:
                    inlet_position = 'end'  # Origin is at geometry end
                    print(f"    BACKWARD flow: origin node {to_node_id} at geometry END -> inlet at END")
                else:
                    # This shouldn't happen if geometry parsing is correct
                    inlet_position = 'end'  # fallback
                    print(
                        f"    BACKWARD flow: WARNING - origin node {to_node_id} not clearly at geometry endpoints -> default END")
            else:
                direction = 'none'
                inlet_position = 'start'  # default
                flow_origin_node = None
                print(f"    NO flow detected -> default inlet at START")

            route_directions[idx] = {
                'direction': direction,
                'inlet_position': inlet_position,
                'from_node': from_node_id,
                'to_node': to_node_id,
                'geometry_start_node': geometry_start_node,
                'geometry_end_node': geometry_end_node,
                'flow_origin_node': flow_origin_node,
                'method': method_used,
                'forward_value': forward_value,
                'backward_value': backward_value
            }

            debug_info.append({
                'idx': idx,
                'from_node': from_node_id,
                'to_node': to_node_id,
                'geometry_start_node': geometry_start_node,
                'geometry_end_node': geometry_end_node,
                'direction': direction,
                'inlet_position': inlet_position,
                'flow_origin_node': flow_origin_node,
                'method': method_used,
                'forward_val': forward_value,
                'backward_val': backward_value
            })

        except Exception as e:
            print(f"Error processing {route_type} route {idx}: {e}")
            route_directions[idx] = {
                'direction': 'error',
                'inlet_position': 'start',
                'from_node': None,
                'to_node': None,
                'geometry_start_node': None,
                'geometry_end_node': None,
                'flow_origin_node': None,
                'method': 'error',
                'forward_value': 0,
                'backward_value': 0
            }

    # Print debug summary
    direction_counts = {}
    method_counts = {}
    for info in debug_info:
        direction = info['direction']
        method = info['method']
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        method_counts[method] = method_counts.get(method, 0) + 1

    print(f"{route_type} direction analysis:")
    for direction, count in direction_counts.items():
        print(f"  {direction}: {count} routes")

    print(f"{route_type} method analysis:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} routes")

    return route_directions


def plot_route_with_inlet_emphasis_fixed(ax, coords, color, direction_info, linewidth, base_alpha):
    """
    FIXED version: Plot route with correctly positioned inlet segment
    """
    direction = direction_info.get('direction', 'unknown')
    inlet_position = direction_info.get('inlet_position', 'start')

    # Debug: Print direction info for first few routes
    static_counter = getattr(plot_route_with_inlet_emphasis_fixed, 'counter', 0)
    if static_counter < 10:  # Print first 10 routes for debugging
        from_node = direction_info.get('from_node')
        to_node = direction_info.get('to_node')
        flow_origin = direction_info.get('flow_origin_node')
        print(
            f"Route {static_counter}: {from_node}->{to_node}, direction={direction}, inlet_position={inlet_position}, flow_origin={flow_origin}, points={len(coords)}")
    plot_route_with_inlet_emphasis_fixed.counter = static_counter + 1

    # Route tails at 35% opacity as requested
    rest_alpha = 0.35

    # Special handling for bidirectional routes - make them visually distinct but not dominant
    if direction == 'bidirectional':
        print(f"    Handling bidirectional route - using DISTINCT visual style")

        if len(coords) == 2:
            # For 2-point bidirectional routes - use a distinctive visual approach
            start_point = coords[0]
            end_point = coords[1]
            total_distance = ((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2) ** 0.5
            target_inlet_distance = 0.15  # ~0.15 degrees

            if total_distance > 2 * target_inlet_distance:
                # Route is long enough for both inlet segments plus middle section
                inlet_ratio = target_inlet_distance / total_distance

                # Calculate split points
                start_split_x = start_point[0] + inlet_ratio * (end_point[0] - start_point[0])
                start_split_y = start_point[1] + inlet_ratio * (end_point[1] - start_point[1])
                start_split_point = (start_split_x, start_split_y)

                end_split_x = start_point[0] + (1 - inlet_ratio) * (end_point[0] - start_point[0])
                end_split_y = start_point[1] + (1 - inlet_ratio) * (end_point[1] - start_point[1])
                end_split_point = (end_split_x, end_split_y)

                # Validate split points are different from endpoints
                min_distance = 0.001
                start_valid = ((start_split_point[0] - start_point[0]) ** 2 + (
                            start_split_point[1] - start_point[1]) ** 2) ** 0.5 > min_distance
                end_valid = ((end_split_point[0] - end_point[0]) ** 2 + (
                            end_split_point[1] - end_point[1]) ** 2) ** 0.5 > min_distance
                middle_valid = ((end_split_point[0] - start_split_point[0]) ** 2 + (
                            end_split_point[1] - start_split_point[1]) ** 2) ** 0.5 > min_distance

                if start_valid and end_valid and middle_valid:
                    # Draw middle segment first with LOW opacity (similar to tails)
                    middle_coords = [start_split_point, end_split_point]
                    middle_line = LineString(middle_coords)
                    gpd.GeoSeries([middle_line]).plot(ax=ax, color=color, linewidth=linewidth, alpha=0.4, zorder=6)

                    # Draw both inlet segments with MEDIUM opacity (60% - clearly less than unidirectional inlets)
                    start_inlet_coords = [start_point, start_split_point]
                    start_inlet_line = LineString(start_inlet_coords)
                    gpd.GeoSeries([start_inlet_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.2, alpha=0.6,
                                                           zorder=7)

                    end_inlet_coords = [end_split_point, end_point]
                    end_inlet_line = LineString(end_inlet_coords)
                    gpd.GeoSeries([end_inlet_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.2, alpha=0.6,
                                                         zorder=7)

                    print(f"    Drew bidirectional: inlets at 60% opacity, middle at 40% opacity")
                else:
                    # Split points too close - draw entire route with medium opacity
                    full_line = LineString([start_point, end_point])
                    gpd.GeoSeries([full_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.1, alpha=0.55, zorder=6)
                    print(f"    Drew entire bidirectional route at 55% opacity")
            else:
                # Route is too short - draw entire route with medium opacity
                full_line = LineString([start_point, end_point])
                gpd.GeoSeries([full_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.1, alpha=0.55, zorder=6)
                print(f"    Drew short bidirectional route at 55% opacity")
        else:
            # Multi-point bidirectional routes - draw with medium opacity
            full_line = LineString(coords)
            gpd.GeoSeries([full_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.1, alpha=0.55, zorder=6)
            print(f"    Drew multi-point bidirectional route at 55% opacity")
        return

    # FIXED: For unidirectional routes, use the inlet_position determined by flow analysis
    inlet_at_start = (inlet_position == 'start')

    # Special handling for 2-point routes (most common case)
    if len(coords) == 2:
        print(f"    Handling 2-point route with inlet_position: {inlet_position}")

        start_point = coords[0]
        end_point = coords[1]

        # Calculate total distance of the 2-point route
        total_distance = ((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2) ** 0.5

        # Use same target inlet distance as multi-point routes
        target_inlet_distance = 0.15  # ~0.15 degrees (about 15-20km)

        # Calculate split ratio based on fixed distance (not percentage)
        if total_distance > target_inlet_distance:
            # Route is longer than target inlet distance
            split_ratio = target_inlet_distance / total_distance
            if not inlet_at_start:
                split_ratio = 1.0 - split_ratio  # Put inlet at end
        else:
            # Route is shorter than target inlet distance - use entire route as inlet
            if inlet_at_start:
                split_ratio = 1.0  # Entire route is inlet
            else:
                split_ratio = 0.0  # Entire route is inlet (at end)

        print(f"    Route distance: {total_distance:.3f} degrees, target inlet: {target_inlet_distance:.3f} degrees")
        print(f"    Split ratio: {split_ratio:.3f}, inlet_at_start: {inlet_at_start}")

        # Only create split if we have a meaningful split (not entire route)
        if 0.05 < split_ratio < 0.95:  # Only split if meaningful (5-95%)
            split_x = start_point[0] + split_ratio * (end_point[0] - start_point[0])
            split_y = start_point[1] + split_ratio * (end_point[1] - start_point[1])
            split_point = (split_x, split_y)

            # Validate split point is sufficiently different from endpoints
            min_distance = 0.001  # Minimum distance in degrees
            start_dist = ((split_point[0] - start_point[0]) ** 2 + (split_point[1] - start_point[1]) ** 2) ** 0.5
            end_dist = ((split_point[0] - end_point[0]) ** 2 + (split_point[1] - end_point[1]) ** 2) ** 0.5

            if start_dist > min_distance and end_dist > min_distance:
                if inlet_at_start:
                    # Draw tail segment first with 35% opacity
                    tail_coords = [split_point, end_point]
                    tail_line = LineString(tail_coords)
                    gpd.GeoSeries([tail_line]).plot(ax=ax, color=color, linewidth=linewidth, alpha=rest_alpha, zorder=7)

                    # Draw inlet segment with full opacity and thicker line
                    inlet_coords = [start_point, split_point]
                    inlet_line = LineString(inlet_coords)
                    gpd.GeoSeries([inlet_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.5, alpha=1.0, zorder=9)
                    print(
                        f"    Drew inlet segment (start {split_ratio * 100:.1f}%) and tail segment (end {(1 - split_ratio) * 100:.1f}%)")
                else:
                    # Draw tail segment first with 35% opacity
                    tail_coords = [start_point, split_point]
                    tail_line = LineString(tail_coords)
                    gpd.GeoSeries([tail_line]).plot(ax=ax, color=color, linewidth=linewidth, alpha=rest_alpha, zorder=7)

                    # Draw inlet segment with full opacity and thicker line
                    inlet_coords = [split_point, end_point]
                    inlet_line = LineString(inlet_coords)
                    gpd.GeoSeries([inlet_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.5, alpha=1.0, zorder=9)
                    print(
                        f"    Drew tail segment (start {split_ratio * 100:.1f}%) and inlet segment (end {(1 - split_ratio) * 100:.1f}%)")
            else:
                # Split point too close to endpoints - draw entire route as inlet
                full_line = LineString([start_point, end_point])
                gpd.GeoSeries([full_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.5, alpha=1.0, zorder=9)
                print(f"    Split point too close - drew entire route as inlet")
        else:
            # Route is too short to split meaningfully - draw entire route as inlet
            full_line = LineString([start_point, end_point])
            gpd.GeoSeries([full_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.5, alpha=1.0, zorder=9)
            print(f"    Route too short to split - drew entire route as inlet")

        return

    # For multi-point routes, implement the SAME logic as 2-point routes
    print(f"    Handling multi-point route with inlet_position: {inlet_position}")

    target_inlet_distance = 0.15  # ~0.15 degrees (about 15-20km)

    # Calculate cumulative distances along the route
    cumulative_distances = [0.0]
    total_distance = 0.0

    for i in range(1, len(coords)):
        segment_distance = ((coords[i][0] - coords[i - 1][0]) ** 2 +
                            (coords[i][1] - coords[i - 1][1]) ** 2) ** 0.5
        total_distance += segment_distance
        cumulative_distances.append(total_distance)

    print(
        f"    Multi-point route distance: {total_distance:.3f} degrees, target inlet: {target_inlet_distance:.3f} degrees")

    # Calculate split ratio based on fixed distance (same as 2-point routes)
    if total_distance > target_inlet_distance:
        # Route is longer than target inlet distance
        split_ratio = target_inlet_distance / total_distance
        if not inlet_at_start:
            split_ratio = 1.0 - split_ratio  # Put inlet at end
    else:
        # Route is shorter than target inlet distance - use entire route as inlet
        if inlet_at_start:
            split_ratio = 1.0  # Entire route is inlet
        else:
            split_ratio = 0.0  # Entire route is inlet (at end)

    print(f"    Multi-point split ratio: {split_ratio:.3f}, inlet_at_start: {inlet_at_start}")

    # Only create split if we have a meaningful split (not entire route)
    if 0.05 < split_ratio < 0.95:  # Only split if meaningful (5-95%)
        # Find the split point based on cumulative distance
        target_split_distance = split_ratio * total_distance
        split_idx = len(coords) - 1  # Default to end

        for i, cum_dist in enumerate(cumulative_distances):
            if cum_dist >= target_split_distance:
                split_idx = max(1, i)  # Ensure at least 1 point for inlet
                break

        split_idx = min(split_idx, len(coords) - 1)  # Don't exceed array bounds

        if inlet_at_start:
            # Draw tail segment first with 35% opacity
            if split_idx < len(coords) - 1:
                tail_coords = coords[split_idx - 1:]  # Overlap by one point for continuity
                if len(tail_coords) >= 2:
                    tail_line = LineString(tail_coords)
                    gpd.GeoSeries([tail_line]).plot(ax=ax, color=color, linewidth=linewidth, alpha=rest_alpha, zorder=7)

            # Draw inlet segment with full opacity and thicker line
            inlet_coords = coords[:split_idx + 1]
            if len(inlet_coords) >= 2:
                inlet_line = LineString(inlet_coords)
                gpd.GeoSeries([inlet_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.5, alpha=1.0, zorder=9)
                print(
                    f"    Drew multi-point inlet segment (start {split_ratio * 100:.1f}%, {len(inlet_coords)} points) and tail segment")
            else:
                # Fallback - entire route as inlet
                full_line = LineString(coords)
                gpd.GeoSeries([full_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.5, alpha=1.0, zorder=9)
                print(f"    Multi-point inlet too small - drew entire route as inlet")
        else:
            # Draw tail segment first with 35% opacity
            if split_idx > 0:
                tail_coords = coords[:split_idx + 1]
                if len(tail_coords) >= 2:
                    tail_line = LineString(tail_coords)
                    gpd.GeoSeries([tail_line]).plot(ax=ax, color=color, linewidth=linewidth, alpha=rest_alpha, zorder=7)

            # Draw inlet segment with full opacity and thicker line
            inlet_coords = coords[split_idx:]
            if len(inlet_coords) >= 2:
                inlet_line = LineString(inlet_coords)
                gpd.GeoSeries([inlet_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.5, alpha=1.0, zorder=9)
                print(
                    f"    Drew multi-point tail segment (start {split_ratio * 100:.1f}%) and inlet segment (end {(1 - split_ratio) * 100:.1f}%, {len(inlet_coords)} points)")
            else:
                # Fallback - entire route as inlet
                full_line = LineString(coords)
                gpd.GeoSeries([full_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.5, alpha=1.0, zorder=9)
                print(f"    Multi-point inlet too small - drew entire route as inlet")
    else:
        # Route is too short to split meaningfully OR split ratio is extreme - draw entire route as inlet
        full_line = LineString(coords)
        gpd.GeoSeries([full_line]).plot(ax=ax, color=color, linewidth=linewidth * 1.5, alpha=1.0, zorder=9)
        print(f"    Multi-point route too short to split (ratio: {split_ratio:.3f}) - drew entire route as inlet")


# Simple route plotting function for overview (no inlet emphasis)
def plot_simple_route(ax, route, color, linewidth=2, alpha=0.7):
    """Simple route plotting without inlet emphasis for overview"""
    line = route.geometry
    gpd.GeoSeries([line]).plot(ax=ax, color=color, linewidth=linewidth, alpha=alpha, zorder=5)


# Enhanced plotting function with FIXED segmented transparency
def plot_route_with_enhanced_direction_fixed(ax, route, color, direction_info, linewidth=4, alpha=0.7, show_inlet=True):
    """FIXED Enhanced plotting with segmented transparency"""
    # Get route coordinates
    line = route.geometry
    coords = list(line.coords)

    if len(coords) < 2:
        return

    if show_inlet:
        # Plot route with FIXED segmented transparency (inlet emphasis)
        plot_route_with_inlet_emphasis_fixed(ax, coords, color, direction_info, linewidth, alpha)
    else:
        # Plot simple route without inlet emphasis
        gpd.GeoSeries([line]).plot(ax=ax, color=color, linewidth=linewidth, alpha=alpha, zorder=5)


# Analyze route directions with FIXED function
print("\nAnalyzing route directions using FIXED logic...")

# Reset debug counter
plot_route_with_inlet_emphasis_fixed.counter = 0

# Use the FIXED analysis logic for all transport modes
pipeline_directions = get_route_directionality_fixed(routes_pipeline, network_pipeline, 'Pipeline')
truck_directions = get_route_directionality_fixed(routes_truck, network_truck, 'Truck')
railway_directions = get_route_directionality_fixed(routes_railway, network_railway, 'Railway')

print("\n" + "=" * 80)
print("FIXED DIRECTION LOGIC FOR ALL TRANSPORT MODES")
print("=" * 80)

# Define northern Italy bounds
north_italy_bounds = {
    'minx': 8.5, 'maxx': 13.0, 'miny': 44.25, 'maxy': 46.0
}

# Create the enhanced plot
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(22, 16))
gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.25, width_ratios=[1.2, 1])

# Left subplot: Overview (simple routes without inlet emphasis)
ax1 = fig.add_subplot(gs[:, 0])
italy.boundary.plot(ax=ax1, color='black', linewidth=1.5, alpha=0.8)
italy.plot(ax=ax1, color='lightgray', alpha=0.3)

# Plot all routes WITHOUT inlet emphasis for overview
for idx, route in routes_pipeline.iterrows():
    plot_simple_route(ax1, route, route_colors['pipeline'], linewidth=2, alpha=1.0)

for idx, route in routes_truck.iterrows():
    plot_simple_route(ax1, route, route_colors['truck'], linewidth=1.5, alpha=1.0)

for idx, route in routes_railway.iterrows():
    plot_simple_route(ax1, route, route_colors['railway'], linewidth=2, alpha=1.0)

# Plot nodes with enhanced visibility
nodes_selected.plot(ax=ax1, color='red', markersize=120, alpha=1.0,
                    edgecolors='white', linewidth=2.5, zorder=20)

# Highlight northern Italy region
from matplotlib.patches import Rectangle

rect = Rectangle((north_italy_bounds['minx'], north_italy_bounds['miny']),
                 north_italy_bounds['maxx'] - north_italy_bounds['minx'],
                 north_italy_bounds['maxy'] - north_italy_bounds['miny'],
                 linewidth=3, edgecolor='red', facecolor='none', linestyle='--')
ax1.add_patch(rect)

ax1.set_title('Italy Transportation Routes Overview\n(Red box shows detailed area)',
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Longitude', fontsize=12)
ax1.set_ylabel('Latitude', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal')

# Create detailed subplots for each transport mode WITH FIXED inlet emphasis
transport_modes = [
    (routes_pipeline, pipeline_directions, 'Pipeline', route_colors['pipeline'], gs[0, 1]),
    (routes_truck, truck_directions, 'Truck', route_colors['truck'], gs[1, 1]),
    (routes_railway, railway_directions, 'Railway', route_colors['railway'], gs[2, 1])
]

for routes, directions, mode_name, color, grid_pos in transport_modes:
    ax = fig.add_subplot(grid_pos)

    # Plot Italy boundary
    italy.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.6)
    italy.plot(ax=ax, color='lightgray', alpha=0.2)

    # Plot routes with FIXED enhanced directional indicators and inlet emphasis
    for idx, route in routes.iterrows():
        plot_route_with_enhanced_direction_fixed(ax, route, color, directions[idx],
                                                 linewidth=4, alpha=1.0, show_inlet=True)

    # Plot nodes with enhanced visibility (smaller and black for detailed views)
    nodes_selected.plot(ax=ax, color='black', markersize=100, alpha=1.0,
                        edgecolors='white', linewidth=2.5, zorder=25)

    # Set bounds and formatting
    ax.set_xlim(north_italy_bounds['minx'], north_italy_bounds['maxx'])
    ax.set_ylim(north_italy_bounds['miny'], north_italy_bounds['maxy'])
    ax.set_title(f'{mode_name} Network - Northern Italy\n({len(routes)} routes)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

# Add simplified legend
legend_elements = [
    plt.Line2D([0], [0], color=route_colors['pipeline'], lw=4, label='Pipeline'),
    plt.Line2D([0], [0], color=route_colors['truck'], lw=4, label='Truck'),
    plt.Line2D([0], [0], color=route_colors['railway'], lw=4, label='Railway'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
               markersize=10, markeredgecolor='white', markeredgewidth=2, label='Network Nodes', linestyle='None'),
    plt.Line2D([0], [0], color='black', lw=6, alpha=1.0, label='Inlet segment (100% opacity)'),
    plt.Line2D([0], [0], color='gray', lw=4, alpha=0.35, label='Route middle/tail (35% opacity)')
]

ax1.legend(handles=legend_elements, loc='lower left', fontsize=10,
           frameon=True, fancybox=True, shadow=True)

# Save the plot
output_filename = "italy_transportation_fixed_inlets.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.2)

print(f"\nPlot saved as: {output_filename}")

# Print comprehensive statistics
print(f"\nFIXED Direction Analysis:")
print("=" * 60)

for mode_name, directions in [('Pipeline', pipeline_directions),
                              ('Truck', truck_directions),
                              ('Railway', railway_directions)]:
    print(f"\n{mode_name} Routes:")
    direction_counts = {}
    method_counts = {}

    for route_info in directions.values():
        direction = route_info['direction']
        method = route_info['method']
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        method_counts[method] = method_counts.get(method, 0) + 1

    print(f"  Direction distribution:")
    for direction, count in direction_counts.items():
        percentage = (count / len(directions)) * 100
        print(f"    {direction}: {count} routes ({percentage:.1f}%)")

    print(f"  Detection methods used:")
    for method, count in method_counts.items():
        print(f"    {method}: {count} routes")

print(f"\nFIXED Visualization Features:")
print("- FIXED: Inlet positioning now based on actual flow origin node location in geometry")
print("- FIXED: Proper handling of cases where Node column order doesn't match geometry direction")
print("- Node column parsing: handles comma ',', dash '-', and space-separated formats")
print("- Matrix flow analysis: forward/backward/bidirectional detection from network matrices")
print("- Geometry validation: identifies which actual nodes are at geometry endpoints")
print("- Smart inlet positioning: inlet appears at the geometry location of the flow origin node")
print("- Forward routes: inlet near from_node (wherever it appears in the route geometry)")
print("- Backward routes: inlet near to_node (wherever it appears in the route geometry)")
print("- Bidirectional routes: inlet segments at BOTH ends (dual inlet design)")
print("- Same target inlet length: ~0.15 degrees (~15-20km) for all routes")
print("- Route middle/tail sections at 35% opacity, inlet segments at 100% opacity + thicker lines")
print("- Proper handling of short routes (entire route becomes inlet if too short to split)")
print("- Enhanced debug output shows flow origin node and its geometry position")
print("- Enhanced node visibility: red nodes in overview, black nodes in detailed views")

plt.show()