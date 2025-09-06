"""
Arc Specific Functions Module

Common functions for CO2 pipeline network analysis, used by both:
- enhanced_co2_pipeline_cost_comparison.py
- full_network_gamma_calculator.py

This module provides data loading, network analysis, and utility functions
for CO2 pipeline cost modeling and optimization.

Updated to use 0 instead of NaN/None for gamma values when calculation fails.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import contextlib
import io

# Import the calculate_annual_emission_values function from the main project
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import functions from the main defined_functions module
try:
    from data_process.utilities.defined_functions import (
        calculate_annual_emission_values,
        update_network_distance_matrix,
        update_network_connection_matrix,
        update_network_size_max_arcs,
    )
except ImportError as e:
    print(f"⚠️  Warning: Could not import some functions from defined_functions: {e}")
    # If import fails, we'll define minimal fallbacks only if needed


# ============================================================================
# OUTPUT SUPPRESSION UTILITIES
# ============================================================================

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily"""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_network_data(data_path="../../northern_italy_data"):
    """
    Load all required network and geographical data

    Args:
        data_path: Path to the data directory

    Returns:
        dict: Dictionary containing all loaded data
    """
    print(f"\n{'=' * 80}")
    print("LOADING NETWORK DATA")
    print(f"{'=' * 80}")

    path_data_case_study = Path(data_path)
    path_files_grids = path_data_case_study / "geographical_feature"
    path_files_node_flux = path_data_case_study / "geographical_feature"
    path_files_electricity = path_data_case_study / "electricity_metrics"

    # Load geographical feature data
    print("Loading geographical data...")
    soil_data = pd.read_csv(path_files_grids / "soil_type_grids_italy.csv")
    anthro_data = pd.read_csv(path_files_grids / "anthropisation_grids_italy.csv")
    morpho_data = pd.read_csv(path_files_grids / "morphological_feature_grids_italy.csv")
    print(
        f"✅ Loaded geographical data: {len(soil_data)} soil grids, {len(anthro_data)} anthro grids, {len(morpho_data)} morpho grids")

    # Load network data - UPDATED TO USE CONFIRMED SHEET NAMES
    print("Loading network data...")
    network_nodes = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0, sheet_name='nodes')
    network_emission_flux = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0, sheet_name='nodes')

    # Load pipeline data (contains both distance and connection info)
    network_pipeline = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0, sheet_name='pipeline')
    network_distance = network_pipeline.copy()  # Use pipeline data as distance matrix
    print(f"✅ Loaded network data: {len(network_nodes)} nodes, {network_distance.shape} distance matrix, {network_pipeline.shape} pipeline matrix")

    # Load electricity data
    print("Loading electricity data...")
    electricity_price = pd.read_csv(path_files_electricity / "electricity_prices_hourly_2024.csv")
    avg_electricity_price_eur_mwh = calculate_average_electricity_price(electricity_price)
    print(f"✅ Average electricity price: {avg_electricity_price_eur_mwh} EUR/MWh")

    # Calculate annual emission values
    print("Processing emission data...")
    network_emission_flux = calculate_annual_emission_values(network_emission_flux)
    print(f"✅ Processed emission data for {len(network_emission_flux)} nodes")

    return {
        'network_nodes': network_nodes,
        'network_emission_flux': network_emission_flux,
        'network_distance': network_distance,
        'network_pipeline': network_pipeline,
        'soil_data': soil_data,
        'anthro_data': anthro_data,
        'morpho_data': morpho_data,
        'electricity_price': electricity_price,
        'avg_electricity_price_eur_mwh': avg_electricity_price_eur_mwh
    }


def load_intersection_data(intersection_file_path, pipeline_names=None):
    """
    Load intersection data for specified pipelines or all available pipelines

    Args:
        intersection_file_path: Path to the intersection Excel file
        pipeline_names: List of pipeline names to load (if None, loads all available)

    Returns:
        dict: Dictionary with intersection data for each pipeline
    """
    print("Loading intersection data...")
    intersection_file = Path(intersection_file_path)
    intersection_data = {}

    if not intersection_file.exists():
        print(f"⚠️  Intersection file not found: {intersection_file}")
        return intersection_data

    try:
        xl_file = pd.ExcelFile(intersection_file)
        available_sheets = xl_file.sheet_names
        print(f"Available intersection sheets: {available_sheets}")

        # If no specific pipeline names provided, use all available sheets
        if pipeline_names is None:
            pipeline_names = available_sheets

        for pipeline_name in pipeline_names:
            if pipeline_name in available_sheets:
                try:
                    pipeline_data = pd.read_excel(intersection_file, sheet_name=pipeline_name)

                    # Look for grid ID and proportion columns
                    grid_col = None
                    prop_col = None

                    for col in pipeline_data.columns:
                        col_lower = str(col).lower()
                        if 'grid' in col_lower and ('id' in col_lower or 'oid' in col_lower):
                            grid_col = col
                        elif 'proportion' in col_lower or 'prop' in col_lower or 'weight' in col_lower:
                            prop_col = col

                    if grid_col is None:
                        grid_col = pipeline_data.columns[0]
                    if prop_col is None and len(pipeline_data.columns) > 1:
                        prop_col = pipeline_data.columns[1]

                    if grid_col and prop_col:
                        # Clean and extract data
                        intersected_grids = []
                        intersected_proportions = []

                        for grid, prop in zip(pipeline_data[grid_col], pipeline_data[prop_col]):
                            if pd.notna(grid) and pd.notna(prop):
                                try:
                                    grid_clean = int(grid) if str(grid).replace('.0', '').isdigit() else grid
                                    prop_clean = float(prop)
                                    intersected_grids.append(grid_clean)
                                    intersected_proportions.append(prop_clean)
                                except (ValueError, TypeError):
                                    continue

                        intersection_data[pipeline_name] = {
                            'intersected_grids': intersected_grids,
                            'intersected_proportions': intersected_proportions
                        }
                        print(f"✅ Loaded intersection data for {pipeline_name}: {len(intersected_grids)} grids")
                    else:
                        print(f"⚠️  Could not identify grid/proportion columns for {pipeline_name}")
                        intersection_data[pipeline_name] = {'intersected_grids': [], 'intersected_proportions': []}

                except Exception as e:
                    print(f"⚠️  Error loading intersection data for {pipeline_name}: {e}")
                    intersection_data[pipeline_name] = {'intersected_grids': [], 'intersected_proportions': []}
            else:
                # No intersection data available for this pipeline
                intersection_data[pipeline_name] = {'intersected_grids': [], 'intersected_proportions': []}

        print(f"✅ Loaded intersection data for {len(intersection_data)} pipelines")

    except Exception as e:
        print(f"❌ Error loading intersection file: {e}")

    return intersection_data


# ============================================================================
# EMISSION AND FLOW CALCULATIONS
# ============================================================================

def get_node_emission(node_id, network_emission_flux):
    """
    Get emission value for a specific node

    Args:
        node_id: Node identifier (can be string or int)
        network_emission_flux: DataFrame with emission data containing 'annual_emission' column

    Returns:
        float: Annual emission in kg/year
    """
    # Check if annual_emission column exists
    if 'annual_emission' not in network_emission_flux.columns:
        print(f"⚠️  Warning: 'annual_emission' column not found in DataFrame")
        print(f"Available columns: {list(network_emission_flux.columns)}")
        return 0.0

    try:
        # Handle both index-based and node_name-based lookups
        if node_id in network_emission_flux.index:
            emission_value = network_emission_flux.loc[node_id, 'annual_emission']
        elif 'node_name' in network_emission_flux.columns:
            # Look up by node_name column
            matching_rows = network_emission_flux[network_emission_flux['node_name'] == node_id]
            if len(matching_rows) > 0:
                emission_value = matching_rows['annual_emission'].sum()  # Sum if multiple rows
            else:
                return 0.0
        else:
            return 0.0

        # Handle pandas Series or array-like objects
        if hasattr(emission_value, 'iloc'):
            emission_value = emission_value.iloc[0] if len(emission_value) > 0 else 0.0

        # Convert to float and handle NaN values
        return float(emission_value) if not pd.isna(emission_value) else 0.0

    except (KeyError, IndexError, ValueError, TypeError) as e:
        print(f"⚠️  Warning: Error getting emission for node {node_id}: {e}")
        return 0.0


def calculate_total_annual_emission(network_emission_flux):
    """
    Calculate total annual emission across all nodes

    Args:
        network_emission_flux: DataFrame with emission data containing 'annual_emission' column

    Returns:
        float: Total annual emission in kg/year
    """
    total_emission = 0.0

    print(f"🔍 Debug: Network emission flux columns: {list(network_emission_flux.columns)}")
    print(f"🔍 Debug: Network emission flux shape: {network_emission_flux.shape}")

    # Check if annual_emission column exists
    if 'annual_emission' not in network_emission_flux.columns:
        print(f"❌ Error: 'annual_emission' column not found!")
        return 0.0

    # Calculate total using vectorized operation for better performance
    try:
        # Option 1: Vectorized calculation (faster)
        valid_emissions = network_emission_flux['annual_emission'].fillna(0.0)
        total_emission = float(valid_emissions.sum())

        # Debug output for individual nodes with significant emissions
        significant_nodes = network_emission_flux[network_emission_flux['annual_emission'] > 0]
        print(f"🔍 Debug: Found {len(significant_nodes)} nodes with emissions")

        # Show some examples for debugging
        if len(significant_nodes) > 0:
            print(f"🔍 Debug: Sample emissions:")
            for i, (node_id, row) in enumerate(significant_nodes.head(5).iterrows()):
                emission_value = row['annual_emission']
                print(f"   Node {node_id}: {emission_value:,.0f} kg/year")

            if len(significant_nodes) > 5:
                print(f"   ... and {len(significant_nodes) - 5} more nodes with emissions")

    except Exception as e:
        print(f"❌ Error in vectorized calculation: {e}")
        print("🔄 Falling back to node-by-node calculation...")

        # Option 2: Fallback to node-by-node calculation
        for node_id in network_emission_flux.index:
            emission_value = get_node_emission(node_id, network_emission_flux)
            if emission_value > 0:
                total_emission += emission_value

    print(f"✅ Total annual emission: {total_emission:,.0f} kg/year")
    return total_emission


def calculate_global_max_massflow(network_emission_flux):
    """
    Calculate global maximum mass flow rate across all nodes

    Args:
        network_emission_flux: DataFrame with emission data

    Returns:
        float: Global max mass flow in kg/s
    """
    total_annual_emission = calculate_total_annual_emission(network_emission_flux)
    seconds_per_year = 365 * 24 * 3600
    global_max_massflow_kg_s = total_annual_emission / seconds_per_year

    print(f"📊 Total annual emission: {total_annual_emission:,.0f} kg/year")
    print(f"📊 Global max mass flow: {global_max_massflow_kg_s:.2f} kg/s")

    return global_max_massflow_kg_s


def calculate_global_min_massflow(network_emission_flux):
    """
    Calculate global minimum mass flow based on minimum node-wise annual emission

    Args:
        network_emission_flux: DataFrame with emission data

    Returns:
        float: Global min mass flow in kg/s based on smallest emitting node
    """
    seconds_per_year = 365.25 * 24 * 3600

    # Get emissions from nodes that actually have emissions (> 0)
    if 'annual_emission' not in network_emission_flux.columns:
        print(f"⚠️  Warning: 'annual_emission' column not found for global min calculation")
        return 1.0  # Fallback

    emitting_nodes = network_emission_flux[network_emission_flux['annual_emission'] > 0]

    if emitting_nodes.empty:
        print(f"⚠️  Warning: No emitting nodes found for global min calculation")
        return 1.0  # Fallback if no emitting nodes found

    # Find minimum annual emission from emitting nodes
    min_emission_kg_year = emitting_nodes['annual_emission'].min()
    min_emission_kg_s = min_emission_kg_year / seconds_per_year

    print(f"📊 Global min mass flow: {min_emission_kg_s:.3f} kg/s (from smallest emitting node: {min_emission_kg_year:,.0f} kg/year)")

    return min_emission_kg_s


# ============================================================================
# NETWORK ANALYSIS FUNCTIONS
# ============================================================================

def get_pipeline_length(from_node, to_node, network_distance):
    """
    Get pipeline length between two nodes

    Args:
        from_node: Source node ID (can be string like "2_1" or individual nodes)
        to_node: Target node ID
        network_distance: Distance matrix DataFrame

    Returns:
        float: Length in km, or None if not found
    """
    try:
        # Handle case where from_node is a pipeline name like "2_1"
        if isinstance(from_node, str) and '_' in from_node:
            parts = from_node.split('_')
            if len(parts) == 2:
                node1, node2 = int(parts[0]), int(parts[1])
                # Try both directions
                length = get_pipeline_length(node1, node2, network_distance)
                if length is not None:
                    return length
                return get_pipeline_length(node2, node1, network_distance)

        # Convert to int if needed
        if isinstance(from_node, str):
            from_node = int(from_node)
        if isinstance(to_node, str):
            to_node = int(to_node)

        # Check if nodes exist in the distance matrix
        distance = None

        if from_node in network_distance.index and to_node in network_distance.columns:
            distance = network_distance.loc[from_node, to_node]
        elif to_node in network_distance.index and from_node in network_distance.columns:
            distance = network_distance.loc[to_node, from_node]

        if pd.isna(distance) or distance == 0:
            return None

        return round(float(distance), 2)

    except Exception as e:
        print(f"   ❌ Error getting pipeline length for {from_node} → {to_node}: {e}")
        return None


def validate_pipeline_transport(from_node, to_node, network_pipeline):
    """
    Check if transport is possible between two nodes

    Args:
        from_node: Source node ID
        to_node: Target node ID
        network_pipeline: Binary transport matrix

    Returns:
        bool: True if transport is possible
    """
    try:
        # Convert to int if needed
        if isinstance(from_node, str):
            from_node = int(from_node)
        if isinstance(to_node, str):
            to_node = int(to_node)

        # Check if transport is possible (non-zero value means possible)
        if from_node in network_pipeline.index and to_node in network_pipeline.columns:
            transport_value = network_pipeline.loc[from_node, to_node]
            return bool(transport_value > 0)  # Non-zero means transport possible
        elif to_node in network_pipeline.index and from_node in network_pipeline.columns:
            transport_value = network_pipeline.loc[to_node, from_node]
            return bool(transport_value > 0)  # Non-zero means transport possible

        return False
    except Exception as e:
        print(f"   ❌ Error validating transport for {from_node} → {to_node}: {e}")
        return False


def get_all_possible_arcs(network_pipeline):
    """
    Get all possible transportation arcs from the network matrix

    Args:
        network_pipeline: Binary transport matrix

    Returns:
        list: List of (from_node, to_node) tuples where transport is possible
    """
    possible_arcs = []

    for from_node in network_pipeline.index:
        for to_node in network_pipeline.columns:
            if validate_pipeline_transport(from_node, to_node, network_pipeline):
                possible_arcs.append((from_node, to_node))

    print(f"Found {len(possible_arcs)} possible arcs in network")
    return possible_arcs


def get_pipeline_directions_and_flows(pipeline_name, network_nodes, network_pipeline,
                                      network_emission_flux, global_max_massflow_kg_s):
    """
    Get all possible directions for a pipeline and calculate mass flows for each direction

    Args:
        pipeline_name: String like "2_1"
        network_nodes: DataFrame with node information
        network_pipeline: Binary matrix indicating transport possibilities
        network_emission_flux: DataFrame with emission data
        global_max_massflow_kg_s: Global maximum mass flow rate

    Returns:
        list: List of direction dictionaries with mass flow data
    """
    try:
        parts = pipeline_name.split('_')
        if len(parts) != 2:
            print(f"   ❌ Invalid pipeline name format: {pipeline_name}")
            return []

        node1, node2 = int(parts[0]), int(parts[1])

        # Check transport possibilities
        can_transport_1_to_2 = validate_pipeline_transport(node1, node2, network_pipeline)
        can_transport_2_to_1 = validate_pipeline_transport(node2, node1, network_pipeline)

        if not (can_transport_1_to_2 or can_transport_2_to_1):
            print(f"   ❌ No transport possible for pipeline {pipeline_name}")
            return []

        # Get emissions for both nodes
        emission_node1 = get_node_emission(node1, network_emission_flux)
        emission_node2 = get_node_emission(node2, network_emission_flux)

        # Convert to kg/s
        seconds_per_year = 365 * 24 * 3600
        emission_node1_kg_s = float(emission_node1) / seconds_per_year
        emission_node2_kg_s = float(emission_node2) / seconds_per_year

        # Create direction configurations
        directions = []

        if can_transport_1_to_2:
            min_flow = emission_node1_kg_s
            max_flow = global_max_massflow_kg_s

            directions.append({
                'direction': f"{node1}_to_{node2}",
                'from_node': node1,
                'to_node': node2,
                'massflow_min_kg_per_s': round(min_flow, 2),
                'massflow_max_kg_per_s': round(max_flow, 2),
                'source_emission_kg_year': emission_node1
            })

        if can_transport_2_to_1:
            min_flow = emission_node2_kg_s
            max_flow = global_max_massflow_kg_s

            directions.append({
                'direction': f"{node2}_to_{node1}",
                'from_node': node2,
                'to_node': node1,
                'massflow_min_kg_per_s': round(min_flow, 2),
                'massflow_max_kg_per_s': round(max_flow, 2),
                'source_emission_kg_year': emission_node2
            })

        return directions

    except Exception as e:
        print(f"   ❌ Error analyzing pipeline {pipeline_name}: {e}")
        return []

# ============================================================================
# GAMMA CALCULATIONS
# ============================================================================

def calculate_arc_gammas(from_node, to_node, data_dict, terrain="Onshore"):
    """Calculate gamma1 and gamma2 for a specific arc

    Args:
        from_node: Source node ID
        to_node: Target node ID
        data_dict: Dictionary containing all network data
        terrain: Terrain type ("Onshore" or "Offshore")

    Returns:
        tuple: (gamma1, gamma2) values, returns (0, 0) if calculation fails
    """

    pipeline_name = f"{from_node}_{to_node}"

    # Get pipeline length using shared function
    length_km = get_pipeline_length(from_node, to_node, data_dict['network_distance'])
    if length_km is None:
        print(f"⚠️  No distance data for arc {from_node} → {to_node}, using gamma values 0")
        return 0, 0

    # Get source emission and calculate mass flow using shared function
    source_emission_kg_year = get_node_emission(from_node, data_dict['network_emission_flux'])

    # Calculate global max mass flow using shared function
    global_max_massflow_kg_s = calculate_global_max_massflow(data_dict['network_emission_flux'])

    # Calculate global min mass flow based on minimum node-wise annual emission
    global_min_massflow_kg_s = calculate_global_min_massflow(data_dict['network_emission_flux'])

    # Set mass flow range with proper handling for transport-only nodes
    seconds_per_year = 365.25 * 24 * 3600
    source_emission_kg_s = source_emission_kg_year / seconds_per_year

    # IMPROVED: Handle transport-only nodes using global minimum from emitting nodes
    if source_emission_kg_s < 0.1:  # Transport-only node (no significant local emissions)
        # For transport nodes, use global minimum mass flow based on smallest emitting node
        massflow_min_kg_s = global_min_massflow_kg_s
        print(f"      🚇 Transport-only node {from_node}: using global min flow of {massflow_min_kg_s:.3f} kg/s")
    else:
        # For emission nodes, use emission-based minimum flow
        massflow_min_kg_s = max(source_emission_kg_s, 0.100)
        print(f"      📍 Emission node {from_node}: using emission-based minimum flow of {massflow_min_kg_s:.3f} kg/s")

    massflow_max_kg_s = global_max_massflow_kg_s

    kg_s_to_t_h = 3600 / 1000  # 3 600 s h-¹  ÷ 1 000 kg t-¹
    massflow_min_t_h = massflow_min_kg_s * kg_s_to_t_h
    massflow_max_t_h = massflow_max_kg_s * kg_s_to_t_h

    # Visual terrain indicator
    terrain_info = "🌊 OFFSHORE" if terrain == "Offshore" else "🏞️ ONSHORE"

    print(f"   Processing arc {from_node} → {to_node}: length={length_km:.3f}km, flow={massflow_min_kg_s:.3f}-{massflow_max_kg_s:.3f}kg/s, {terrain_info}")

    # Create base options using shared function with specified terrain
    base_options = create_base_options(
        length_km,
        massflow_min_t_h,
        massflow_max_t_h,
        data_dict['avg_electricity_price_eur_mwh'],
        terrain=terrain,
        evaluation_points=10
    )

    # Add geographical data using shared function
    if pipeline_name in data_dict['intersection_data']:
        intersected_grids = data_dict['intersection_data'][pipeline_name]['intersected_grids']
        intersected_proportions = data_dict['intersection_data'][pipeline_name]['intersected_proportions']
    else:
        intersected_grids = []
        intersected_proportions = []

    enhanced_options = add_geographical_options(
        base_options,
        data_dict['morpho_data'],
        data_dict['soil_data'],
        data_dict['anthro_data'],
        intersected_grids,
        intersected_proportions
    )

    # Calculate costs with enhanced model - SUPPRESS VERBOSE OUTPUT
    try:
        # Import the enhanced model here to avoid circular imports
        from adopt_net0.database.components.networks.enhanced_co2_pipelines_cost_model import \
            CO2_Pipeline_CostModel as EnhancedModel

        model_enhanced = EnhancedModel("CO2_Pipeline")

        # Suppress the verbose output from the cost model
        with suppress_stdout():
            results_enhanced = model_enhanced.calculate_indicators(enhanced_options)

        # IMPROVED: Better error handling for missing keys
        try:
            gamma1 = results_enhanced['financial_indicators']['gamma1']
            gamma2 = results_enhanced['financial_indicators']['gamma2']

            print(f"      ✅ γ₁: {gamma1:,.0f} EUR, γ₂: {gamma2:,.3f} EUR/(t/h)")
            return gamma1, gamma2

        except KeyError as ke:
            print(f"      ❌ Missing result key for arc {from_node} → {to_node}: {ke}, using gamma values 0")
            print(f"      🔍 Available keys: {list(results_enhanced.get('financial_indicators', {}).keys())}")
            return 0, 0

    except Exception as e:
        print(f"      ❌ Error calculating costs for arc {from_node} → {to_node}: {e}, using gamma values 0")

        # IMPROVED: Additional debugging for specific error types
        if "'capex_pipe'" in str(e):
            print(f"      🔍 CAPEX calculation failed - likely due to:")
            print(f"         • Mass flow range: {massflow_min_t_h:.3f} - {massflow_max_t_h:.3f} kg/s")
            print(f"         • Terrain type: {terrain}")
            print(f"         • Pipeline length: {length_km:.3f} km")
            print(f"         • Source emission: {source_emission_kg_year:,.0f} kg/year")

        return 0, 0


def create_gamma_matrices(data_dict, terrain_function=None):
    """Create gamma1 and gamma2 matrices for all arcs

    Args:
        data_dict: Dictionary containing all network data
        terrain_function: Optional function to determine terrain for each arc.
                         Should accept (from_node, to_node) and return terrain string.
                         If None, defaults to "Onshore" for all arcs.

    Returns:
        tuple: (gamma1_matrix, gamma2_matrix)
    """

    print_section_header("CALCULATING GAMMA VALUES FOR ALL ARCS")

    # Get all unique nodes
    all_nodes = sorted(set(data_dict['network_pipeline'].index) | set(data_dict['network_pipeline'].columns))
    print(f"Processing {len(all_nodes)} nodes: {all_nodes}")

    # Initialize matrices with 0 instead of NaN
    gamma1_matrix = pd.DataFrame(index=all_nodes, columns=all_nodes, dtype=float)
    gamma2_matrix = pd.DataFrame(index=all_nodes, columns=all_nodes, dtype=float)

    # Fill with zeros initially
    gamma1_matrix = gamma1_matrix.fillna(0.0)
    gamma2_matrix = gamma2_matrix.fillna(0.0)

    # Get all possible arcs using shared function
    possible_arcs = get_all_possible_arcs(data_dict['network_pipeline'])

    # Fill matrices with gamma values
    processed_arcs = 0

    for from_node, to_node in possible_arcs:
        # Determine terrain for this arc
        if terrain_function is not None:
            terrain = terrain_function(from_node, to_node)
        else:
            terrain = "Onshore"  # Default terrain

        gamma1, gamma2 = calculate_arc_gammas(from_node, to_node, data_dict, terrain=terrain)

        # Always set the values (now they're either calculated values or 0)
        gamma1_matrix.loc[from_node, to_node] = gamma1
        gamma2_matrix.loc[from_node, to_node] = gamma2

        if gamma1 > 0 and gamma2 > 0:
            processed_arcs += 1

    print(f"\n📊 SUMMARY:")
    print(f"   Total possible arcs: {len(possible_arcs)}")
    print(f"   Successfully processed: {processed_arcs}")
    print(f"   Set to zero: {len(possible_arcs) - processed_arcs}")

    return gamma1_matrix, gamma2_matrix


def create_zero_gamma_matrices(gamma1_matrix):
    """
    Create gamma3 and gamma4 matrices with same dimensions as gamma1, filled with zeros

    Args:
        gamma1_matrix: Reference matrix for dimensions

    Returns:
        tuple: (gamma3_matrix, gamma4_matrix) both filled with zeros
    """
    print("Creating gamma3 and gamma4 matrices filled with zeros...")

    # Create matrices with same index and columns as gamma1, filled with zeros
    gamma3_matrix = pd.DataFrame(
        data=0.0,
        index=gamma1_matrix.index,
        columns=gamma1_matrix.columns,
        dtype=float
    )

    gamma4_matrix = pd.DataFrame(
        data=0.0,
        index=gamma1_matrix.index,
        columns=gamma1_matrix.columns,
        dtype=float
    )

    print(f"✅ Created gamma3 matrix: {gamma3_matrix.shape}")
    print(f"✅ Created gamma4 matrix: {gamma4_matrix.shape}")

    return gamma3_matrix, gamma4_matrix

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_average_electricity_price(electricity_price_df):
    """
    Calculate the average electricity price from hourly data
    Uses similar logic to the main script's electricity price processing

    Args:
        electricity_price_df: DataFrame with electricity price data

    Returns:
        float: Average electricity price in EUR/MWh
    """
    print(f"\n🔌 CALCULATING AVERAGE ELECTRICITY PRICE")
    print(f"{'=' * 50}")

    # Find the price column - same logic as main script
    price_column = None
    if 'Day-ahead Price (EUR/MWh)' in electricity_price_df.columns:
        price_column = 'Day-ahead Price (EUR/MWh)'
    else:
        # Try to find a price column with different name
        price_columns = [col for col in electricity_price_df.columns if 'price' in col.lower()]
        if price_columns:
            price_column = price_columns[0]

    if price_column is None:
        print(f"⚠️  Could not identify electricity price column, using default 60.0 EUR/MWh")
        return 60.0

    print(f"Using price column: '{price_column}'")

    # Extract price data
    prices = electricity_price_df[price_column].copy()

    # Handle leap year data (8784 -> 8760) - simplified version of main script logic
    if len(prices) == 8784:
        print("Detected leap year data (8784 hours). Truncating to 8760 hours.")
        prices = prices[:8760]  # Simple truncation approach
        print(f"Reduced to {len(prices)} hours")

    # Clean and process prices
    prices = pd.to_numeric(prices, errors='coerce').dropna()

    if len(prices) == 0:
        print(f"⚠️  No valid price data found, using default 60.0 EUR/MWh")
        return 60.0

    # Calculate average price
    avg_price = prices.mean()
    print(f"📊 Average price: {avg_price:.2f} EUR/MWh")

    # Validate price range
    if 20 <= avg_price <= 200:
        print(f"✅ Average price appears reasonable for European electricity market")
    else:
        print(f"⚠️  Average price outside typical range (20-200 EUR/MWh) - please verify data")

    return round(avg_price, 2)


def create_base_options(length_km, massflow_min_t_h, massflow_max_t_h,
                        avg_electricity_price_eur_mwh, terrain="Onshore", evaluation_points=10):
    """
    Create base options dictionary for cost model calculations

    Args:
        length_km: Pipeline length in km
        massflow_min_t_h: Minimum mass flow in t/h
        massflow_max_t_h: Maximum mass flow in t/h
        avg_electricity_price_eur_mwh: Average electricity price
        terrain: Terrain type ("Onshore" or "Offshore")
        evaluation_points: Number of evaluation points

    Returns:
        dict: Base options for cost model
    """
    return {
        "length_km": length_km,
        "currency_out": "EUR",
        "financial_year_out": 2024,
        "discount_rate": 0.1,
        "massflow_min_t_h": massflow_min_t_h,
        "massflow_max_t_h": massflow_max_t_h,
        "massflow_evaluation_points": evaluation_points,
        "terrain": terrain,
        "timeframe": "mid-term",
        "electricity_price_eur_per_mw": avg_electricity_price_eur_mwh
    }


def add_geographical_options(base_options, morpho_data, soil_data, anthro_data,
                             intersected_grids, intersected_proportions):
    """
    Add geographical data to base options

    Args:
        base_options: Base options dictionary
        morpho_data: Morphological data DataFrame
        soil_data: Soil data DataFrame
        anthro_data: Anthropization data DataFrame
        intersected_grids: List of intersected grid IDs
        intersected_proportions: List of intersection proportions

    Returns:
        dict: Enhanced options with geographical data
    """
    enhanced_options = base_options.copy()
    enhanced_options.update({
        "morpho_data": morpho_data,
        "soil_data": soil_data,
        "anthro_data": anthro_data,
        "intersected_grids": intersected_grids,
        "intersected_proportions": intersected_proportions
    })
    return enhanced_options


def save_to_excel(gamma1_matrix, gamma2_matrix, gamma3_matrix=None, gamma4_matrix=None,
                  filename="capex_defined_per_arc.xlsx"):
    """Save gamma matrices to Excel file with separate sheets

    Args:
        gamma1_matrix: First gamma matrix (calculated values)
        gamma2_matrix: Second gamma matrix (calculated values)
        gamma3_matrix: Third gamma matrix (optional, defaults to zeros if not provided)
        gamma4_matrix: Fourth gamma matrix (optional, defaults to zeros if not provided)
        filename: Output Excel filename
    """

    print_section_header("SAVING RESULTS TO EXCEL")

    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Save gamma1 matrix
            gamma1_matrix.to_excel(writer, sheet_name='gamma1', index=True)
            print(f"✅ Saved gamma1 matrix to sheet 'gamma1'")

            # Save gamma2 matrix
            gamma2_matrix.to_excel(writer, sheet_name='gamma2', index=True)
            print(f"✅ Saved gamma2 matrix to sheet 'gamma2'")

            # Save gamma3 matrix (create with zeros if not provided)
            if gamma3_matrix is None:
                gamma3_matrix = pd.DataFrame(
                    data=0.0,
                    index=gamma1_matrix.index,
                    columns=gamma1_matrix.columns,
                    dtype=float
                )
            gamma3_matrix.to_excel(writer, sheet_name='gamma3', index=True)
            print(f"✅ Saved gamma3 matrix to sheet 'gamma3'")

            # Save gamma4 matrix (create with zeros if not provided)
            if gamma4_matrix is None:
                gamma4_matrix = pd.DataFrame(
                    data=0.0,
                    index=gamma1_matrix.index,
                    columns=gamma1_matrix.columns,
                    dtype=float
                )
            gamma4_matrix.to_excel(writer, sheet_name='gamma4', index=True)
            print(f"✅ Saved gamma4 matrix to sheet 'gamma4'")

        print(f"📁 Excel file saved as: {filename}")
        print(f"   Matrix size: {gamma1_matrix.shape[0]} × {gamma1_matrix.shape[1]}")
        print(f"   Gamma1 non-zero values: {(gamma1_matrix != 0).sum().sum()}")
        print(f"   Gamma2 non-zero values: {(gamma2_matrix != 0).sum().sum()}")
        print(f"   Gamma3 values: {gamma3_matrix.count().sum()} (all zeros)")
        print(f"   Gamma4 values: {gamma4_matrix.count().sum()} (all zeros)")

    except Exception as e:
        print(f"❌ Error saving to Excel: {e}")


def print_section_header(title, width=80):
    """Print a formatted section header"""
    print(f"\n{'=' * width}")
    print(f"{title}")
    print(f"{'=' * width}")


def determine_arc_terrain(from_node, to_node):
    """
    Determine the terrain type for specific arcs in this case study.

    This function contains case-specific logic for the Northern Italy study:
    - Arc 13-14 (in either direction) is offshore
    - All other arcs are onshore

    Args:
        from_node: Source node ID
        to_node: Target node ID

    Returns:
        str: "Offshore" for arc 13-14, "Onshore" for all others
    """
    try:
        # Convert to int if needed for comparison
        from_node_int = int(from_node)
        to_node_int = int(to_node)

        # Check if this is the offshore arc (13-14 in either direction)
        if (from_node_int == 13 and to_node_int == 14) or (from_node_int == 14 and to_node_int == 13):
            return "Offshore"
        else:
            return "Onshore"

    except (ValueError, TypeError):
        # If conversion fails, default to onshore
        print(f"⚠️  Warning: Could not parse node IDs {from_node}, {to_node}. Defaulting to Onshore.")
        return "Onshore"