import pandas as pd
import numpy as np


def analyze_route_grid_intersections(route_gdf, fishnet_gdf):
    """
    Analyze intersections between route lines and fishnet grids.

    Parameters:
    -----------
    route_gdf : GeoDataFrame
        GeoDataFrame containing route geometries
    fishnet_gdf : GeoDataFrame
        GeoDataFrame containing fishnet grid geometries

    Returns:
    --------
    dict : Dictionary with route names as keys and DataFrames as values
    """

    print("\nStarting intersection analysis...")
    results = {}

    for idx, route_row in route_gdf.iterrows():
        route_name = route_row['Name']
        route_geom = route_row.geometry
        route_total_length = route_geom.length

        print(f"Processing route: {route_name[:50]}...")  # Truncate long names for display

        # Find intersecting grids
        intersecting_grids = fishnet_gdf[fishnet_gdf.intersects(route_geom)]

        if len(intersecting_grids) == 0:
            print(f"  No intersections found")
            continue

        intersection_data = []

        for _, grid_row in intersecting_grids.iterrows():
            grid_id = grid_row['grid_id']
            grid_geom = grid_row.geometry

            # Calculate intersection
            intersection = route_geom.intersection(grid_geom)

            if intersection.is_empty:
                continue

            # Calculate length of intersection
            if intersection.geom_type == 'LineString':
                intersection_length = intersection.length
            elif intersection.geom_type == 'MultiLineString':
                intersection_length = sum(line.length for line in intersection.geoms)
            else:
                # Handle cases where intersection might be points or other geometries
                intersection_length = 0

            # Calculate proportion
            proportion = intersection_length / route_total_length if route_total_length > 0 else 0

            intersection_data.append({
                'grid_id': grid_id,
                'intersection_length': intersection_length,
                'proportion': proportion
            })

        # Create DataFrame for this route
        if intersection_data:
            route_df = pd.DataFrame(intersection_data)
            # Sort by grid ID
            route_df = route_df.sort_values('grid_id')
            results[route_name] = route_df

            print(f"  Found {len(route_df)} intersecting grids")
            print(f"  Total proportion: {route_df['proportion'].sum():.4f}")

    return results


def create_node_name_to_id_mapping(nodes_df):
    """
    Create a mapping from node names to node IDs.

    Parameters:
    -----------
    nodes_df : DataFrame
        DataFrame containing node information with 'node_name' and index as node_id

    Returns:
    --------
    dict : Dictionary mapping node names to node IDs
    """
    # Create mapping, handling potential duplicates by taking the first occurrence
    name_to_id = {}
    for node_id, row in nodes_df.iterrows():
        node_name = row['node_name']
        if node_name not in name_to_id:
            name_to_id[node_name] = node_id

    return name_to_id


def convert_route_name_to_node_ids(route_name, name_to_id_mapping):
    """
    Convert route name (format: "NodeName1 - NodeName2") to node ID format ("ID1_ID2").

    Parameters:
    -----------
    route_name : str
        Route name in format "NodeName1 - NodeName2"
    name_to_id_mapping : dict
        Dictionary mapping node names to node IDs

    Returns:
    --------
    str : Sheet name in format "ID1_ID2" or original name if conversion fails
    """
    try:
        print(f"    Converting route: '{route_name}'")

        # Split the route name by " - "
        parts = route_name.split(' - ')

        if len(parts) != 2:
            print(f"      Warning: Route name doesn't follow expected format (found {len(parts)} parts)")
            return route_name.replace(' ', '_')[:31]  # Fallback to modified original name

        node1_name, node2_name = parts[0].strip(), parts[1].strip()
        print(f"      Node 1: '{node1_name}'")
        print(f"      Node 2: '{node2_name}'")

        # Look up node IDs (try exact match first, then partial match)
        node1_id = name_to_id_mapping.get(node1_name)
        node2_id = name_to_id_mapping.get(node2_name)

        # If exact match fails, try partial matching
        if node1_id is None:
            print(f"      Exact match failed for '{node1_name}', trying partial match...")
            for name, node_id in name_to_id_mapping.items():
                if node1_name in name or name in node1_name:
                    node1_id = node_id
                    print(f"      Partial match found: '{name}' → {node_id}")
                    break

        if node2_id is None:
            print(f"      Exact match failed for '{node2_name}', trying partial match...")
            for name, node_id in name_to_id_mapping.items():
                if node2_name in name or name in node2_name:
                    node2_id = node_id
                    print(f"      Partial match found: '{name}' → {node_id}")
                    break

        if node1_id is None:
            print(f"      Warning: Node '{node1_name}' not found in mapping")
            print(f"      Available node names: {list(name_to_id_mapping.keys())[:5]}...")
            return route_name.replace(' ', '_').replace('-', '_')[:31]

        if node2_id is None:
            print(f"      Warning: Node '{node2_name}' not found in mapping")
            print(f"      Available node names: {list(name_to_id_mapping.keys())[:5]}...")
            return route_name.replace(' ', '_').replace('-', '_')[:31]

        result = f"{node1_id}_{node2_id}"
        print(f"      Success: '{route_name}' → '{result}'")
        return result

    except Exception as e:
        print(f"      Error converting route name '{route_name}': {e}")
        return route_name.replace(' ', '_').replace('-', '_')[:31]


def export_to_excel(results_dict, output_path, nodes_df=None):
    """
    Export intersection results to Excel file with separate sheets for each route.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with route names as keys and DataFrames as values
    output_path : str or Path
        Path where the Excel file will be saved
    nodes_df : DataFrame, optional
        DataFrame containing node information for creating node ID-based sheet names
    """

    if not results_dict:
        print("No intersection results to export.")
        return

    print(f"\nExporting results to Excel...")

    # Create node name to ID mapping if nodes_df is provided
    name_to_id_mapping = None
    if nodes_df is not None:
        name_to_id_mapping = create_node_name_to_id_mapping(nodes_df)
        print(f"  Created mapping for {len(name_to_id_mapping)} nodes")

        # Debug: print some of the mapping
        print("  Sample node name to ID mapping:")
        for i, (name, node_id) in enumerate(list(name_to_id_mapping.items())[:5]):
            print(f"    '{name}' → {node_id}")

        # Debug: print a sample route name
        sample_route = list(results_dict.keys())[0]
        print(f"  Sample route name: '{sample_route}'")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for route_name, route_df in results_dict.items():

            # Determine sheet name
            if name_to_id_mapping is not None:
                # Use node IDs for sheet name
                print(f"  Converting route to node IDs...")
                sheet_name = convert_route_name_to_node_ids(route_name, name_to_id_mapping)
            else:
                print(f"  No node mapping available, using original method...")
                # Clean sheet name for Excel compatibility (original method)
                # Excel sheet names have max 31 characters and cannot contain: [ ] * ? : \ /
                sheet_name = str(route_name)[:31]
                invalid_chars = ['[', ']', '*', '?', ':', '\\', '/']
                for char in invalid_chars:
                    sheet_name = sheet_name.replace(char, '_')

            # Handle duplicate sheet names by adding numbers
            original_sheet_name = sheet_name
            counter = 1
            existing_sheets = writer.sheets.keys() if hasattr(writer, 'sheets') else []

            while sheet_name in existing_sheets:
                # Truncate further if needed to fit counter
                max_base_length = 31 - len(f"_{counter}")
                base_name = original_sheet_name[:max_base_length]
                sheet_name = f"{base_name}_{counter}"
                counter += 1

            # Add route name as metadata if using node IDs
            route_df_with_metadata = route_df.copy()
            if name_to_id_mapping is not None:
                # Add a row at the top with the full route name for reference
                metadata_row = pd.DataFrame({
                    'grid_id': [f'Route: {route_name}'],
                    'intersection_length': [''],
                    'proportion': ['']
                })
                route_df_with_metadata = pd.concat([metadata_row, route_df_with_metadata],
                                                   ignore_index=True)

            route_df_with_metadata.to_excel(writer, sheet_name=sheet_name, index=False)

            # Print export info
            if name_to_id_mapping is not None:
                print(f"  Exported: '{route_name[:40]}...' → sheet '{sheet_name}' (Node IDs)")
            else:
                print(f"  Exported: '{route_name[:40]}...' → sheet '{sheet_name}'")

    print(f"\nAll results exported to: {output_path}")


def print_summary_statistics(intersection_results, fishnet):
    """
    Print comprehensive summary statistics of the intersection analysis.

    Parameters:
    -----------
    intersection_results : dict
        Dictionary with route names as keys and DataFrames as values
    fishnet : GeoDataFrame
        Original fishnet GeoDataFrame for grid count reference
    """

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if intersection_results:
        total_routes = len(intersection_results)
        total_intersections = sum(len(df) for df in intersection_results.values())

        print(f"Total routes processed: {total_routes}")
        print(f"Total grid intersections found: {total_intersections}")
        print(f"Average intersections per route: {total_intersections / total_routes:.2f}")

        # Route with most intersections
        max_intersections_route = max(intersection_results.items(), key=lambda x: len(x[1]))
        print(
            f"Route with most intersections: '{max_intersections_route[0][:40]}...' ({len(max_intersections_route[1])} grids)")

        # Route with least intersections
        min_intersections_route = min(intersection_results.items(), key=lambda x: len(x[1]))
        print(
            f"Route with least intersections: '{min_intersections_route[0][:40]}...' ({len(min_intersections_route[1])} grids)")

        # Proportion validation (should sum to ~1.0 for each route)
        print(f"\nProportion validation (should sum to ~1.0 for each route):")
        proportion_issues = []

        for route_name, df in intersection_results.items():
            prop_sum = df['proportion'].sum()
            if abs(prop_sum - 1.0) > 0.01:  # More than 1% difference
                proportion_issues.append((route_name, prop_sum))

        if proportion_issues:
            print(f"⚠️  Found {len(proportion_issues)} routes with proportion sum issues:")
            for route_name, prop_sum in proportion_issues[:5]:  # Show first 5
                print(f"  '{route_name[:40]}...': {prop_sum:.4f}")
            if len(proportion_issues) > 5:
                print(f"  ... and {len(proportion_issues) - 5} more")
        else:
            print("✓ All routes have proportion sums close to 1.0")

        # Grid usage statistics
        all_grids_used = set()
        for df in intersection_results.values():
            all_grids_used.update(df['grid_id'].tolist())

        print(f"\nGrid usage statistics:")
        print(f"Total unique grids intersected: {len(all_grids_used)}")
        print(f"Percentage of total grids used: {len(all_grids_used) / len(fishnet) * 100:.1f}%")

        # Intersection length statistics
        all_lengths = []
        all_proportions = []
        for df in intersection_results.values():
            all_lengths.extend(df['intersection_length'].tolist())
            all_proportions.extend(df['proportion'].tolist())

        print(f"\nIntersection statistics:")
        print(f"Average intersection length: {np.mean(all_lengths):.6f} degrees")
        print(f"Average intersection proportion: {np.mean(all_proportions):.4f}")
        print(f"Max intersection proportion: {np.max(all_proportions):.4f}")
        print(f"Min intersection proportion: {np.min(all_proportions):.6f}")

    else:
        print("❌ No intersection results found!")
        print("Possible issues:")
        print("- Routes and grids may not spatially overlap")
        print("- Check coordinate reference systems")
        print("- Verify geometry validity")