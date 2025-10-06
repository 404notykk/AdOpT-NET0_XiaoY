import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# Import Crameri color palette for consistency
try:
    import cmcrameri.cm as cmc

    navia_colors = cmc.navia(np.linspace(0.1, 0.9, 8))  # Get colors from navia palette
    ORIGINAL_COLOR = navia_colors[1]  # Dark blue-ish
    ENHANCED_COLOR = navia_colors[6]  # Orange-ish
    DIFF_COLOR = navia_colors[4]  # Green-ish
    REL_DIFF_COLOR = navia_colors[3]  # Purple-ish
    GEO_COLOR = navia_colors[5]  # Yellow-ish
except ImportError:
    print("Warning: cmcrameri not available, using default colors")
    # Fallback colors that approximate navia palette
    ORIGINAL_COLOR = '#2E4372'  # Dark blue
    ENHANCED_COLOR = '#E17A47'  # Orange
    DIFF_COLOR = '#4A7C59'  # Green
    REL_DIFF_COLOR = '#8E4162'  # Purple
    GEO_COLOR = '#D4A574'  # Yellow-orange

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utilities/')))

from adopt_net0.database.components.networks.enhanced_co2_pipelines_cost_model import \
    CO2_Pipeline_CostModel as EnhancedModel
from adopt_net0.database.components.networks.co2_pipelines_cost_model import CO2_Pipeline_CostModel as OriginalModel

from southern_europe.data_process.utilities.defined_functions import calculate_annual_emission_values

# ----- Data loading section -----#

path_data_case_study = Path("../../northern_italy_data")

path_files_grids = path_data_case_study / "geographical_feature"
path_files_node_flux = path_data_case_study / "geographical_feature"
path_files_electricity = path_data_case_study / "electricity_metrics"

# Load geographical feature data
soil_data = pd.read_csv(path_files_grids / "soil_type_grids_italy.csv")
anthro_data = pd.read_csv(path_files_grids / "anthropisation_grids_italy.csv")
morpho_data = pd.read_csv(path_files_grids / "morphological_feature_grids_italy.csv")

# Load network data
network_nodes = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0, sheet_name='nodes')
network_emission_flux = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0,
                                      sheet_name='nodes')  # annual emission fluxes
# Updated: Load single pipeline sheet that contains both connections and distances
network_pipeline = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0, sheet_name='pipeline')

# Load electricity data
electricity_price = pd.read_csv(path_files_electricity / "electricity_prices_hourly_2024.csv")  # electricity price

# Load intersection data - UPDATED: For pipelines 5_6, 13_14, and 1_11
intersection_file = path_files_node_flux / "route_grid_intersections.xlsx"
pipeline_names = ['5_6', '13_14', '1_11']  # UPDATED: Analyze all three pipelines
intersection_data = {}

# Load intersection data for each pipeline
for pipeline_name in pipeline_names:
    try:
        pipeline_data = pd.read_excel(intersection_file, sheet_name=pipeline_name)

        # Look for grid ID column (try different possible names)
        grid_col = None
        prop_col = None

        for col in pipeline_data.columns:
            col_lower = str(col).lower()
            if 'grid' in col_lower and ('id' in col_lower or 'oid' in col_lower):
                grid_col = col
            elif 'proportion' in col_lower or 'prop' in col_lower or 'weight' in col_lower:
                prop_col = col

        if grid_col is None:
            # Try first column as grid ID
            grid_col = pipeline_data.columns[0]

        if prop_col is None:
            # Try looking for numeric columns that could be proportions
            numeric_cols = pipeline_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != grid_col:  # Don't use the same column for both
                    prop_col = col
                    break

            # If still not found, try second column
            if prop_col is None and len(pipeline_data.columns) > 1:
                prop_col = pipeline_data.columns[1]
            else:
                print(f"Warning: No proportion column found for pipeline {pipeline_name}")
                continue

        # Extract grid IDs and proportions
        intersected_grids = pipeline_data[grid_col].dropna().tolist()
        intersected_proportions = pipeline_data[prop_col].dropna().tolist()

        intersection_data[pipeline_name] = {
            'intersected_grids': intersected_grids,
            'intersected_proportions': intersected_proportions
        }

        print(
            f"✅ Loaded intersection data for pipeline {pipeline_name}: {len(intersected_grids)} grids using columns '{grid_col}' and '{prop_col}'")

    except Exception as e:
        print(f"Warning: Could not load intersection data for pipeline {pipeline_name}: {e}")
        intersection_data[pipeline_name] = {
            'intersected_grids': [],
            'intersected_proportions': []
        }


# ----- Electricity price calculation -----#
def calculate_average_electricity_price(electricity_price_df):
    """
    Calculate the average electricity price from the hourly data

    Args:
        electricity_price_df: DataFrame with electricity price data

    Returns:
        float: Average electricity price in EUR/MWh
    """
    print(f"\n🔌 CALCULATING AVERAGE ELECTRICITY PRICE")
    print(f"{'=' * 50}")

    # Find the price column (should be something like "Day-ahead Price (EUR/MWh)")
    price_column = None
    for col in electricity_price_df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['price', 'eur', 'mwh']):
            price_column = col
            break

    if price_column is None:
        raise ValueError("Could not identify electricity price column")

    print(f"Using price column: '{price_column}'")

    # Extract price data and clean it
    prices = electricity_price_df[price_column].copy()
    prices = pd.to_numeric(prices, errors='coerce').dropna()

    # Calculate average price
    avg_price = prices.mean()
    print(f"📊 Average price: {avg_price:.2f} EUR/MWh")

    # Check if the average is reasonable (typical range: 20-200 EUR/MWh)
    if 20 <= avg_price <= 200:
        print(f"✅ Average price appears reasonable for European electricity market")
    else:
        print(f"⚠️  Average price outside typical range (20-200 EUR/MWh) - please verify data")

    return round(avg_price, 2)


# Calculate the average electricity price
try:
    avg_electricity_price_eur_mwh = calculate_average_electricity_price(electricity_price)
    print(f"\n💡 Will use electricity price: {avg_electricity_price_eur_mwh} EUR/MWh")
except Exception as e:
    print(f"❌ Error calculating electricity price: {e}")
    print(f"Using default value of 60.0 EUR/MWh")
    avg_electricity_price_eur_mwh = 60.0

# ----- Emission calculation and mass flow determination -----#

# Calculate the actual annual emission values using the Excel formula logic
network_emission_flux = calculate_annual_emission_values(network_emission_flux)

# Debug: Print information about the network_emission_flux DataFrame
print(f"\n🔍 Debug: Network emission flux info:")
print(f"  Shape: {network_emission_flux.shape}")
print(f"  Index: {list(network_emission_flux.index)}")
print(f"  Columns: {list(network_emission_flux.columns)}")
print(f"  Data types: {network_emission_flux.dtypes}")
if not network_emission_flux.empty:
    print(f"  Sample data:\n{network_emission_flux.head()}")
print()


# Calculate total annual emission to determine global max mass flow
def calculate_total_annual_emission(network_emission_flux):
    """Calculate total annual emission across all nodes"""
    total_emission = 0

    print(f"🔍 Debug: Network emission flux columns: {list(network_emission_flux.columns)}")
    print(f"🔍 Debug: Network emission flux shape: {network_emission_flux.shape}")

    for node_id in network_emission_flux.index:
        # Try different possible column names for emissions
        possible_cols = []
        for col in network_emission_flux.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['emission', 'annual', 'co2', 'flux']):
                possible_cols.append(col)

        emission_value = 0  # Default value

        if possible_cols:
            try:
                emission_value = network_emission_flux.loc[node_id, possible_cols[0]]
                # Handle case where emission_value might be a Series
                if hasattr(emission_value, 'iloc'):
                    emission_value = emission_value.iloc[0] if len(emission_value) > 0 else 0
            except (KeyError, IndexError):
                emission_value = 0
        else:
            # Fallback to first numeric column
            numeric_cols = network_emission_flux.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                try:
                    emission_value = network_emission_flux.loc[node_id, numeric_cols[0]]
                    # Handle case where emission_value might be a Series
                    if hasattr(emission_value, 'iloc'):
                        emission_value = emission_value.iloc[0] if len(emission_value) > 0 else 0
                except (KeyError, IndexError):
                    emission_value = 0

        # Convert to scalar and check if valid
        try:
            emission_scalar = float(emission_value) if not pd.isna(emission_value) else 0
            if emission_scalar > 0:
                total_emission += emission_scalar
                print(f"🔍 Debug: Node {node_id} emission: {emission_scalar:,.0f} kg/year")
        except (ValueError, TypeError):
            print(f"⚠️  Could not convert emission value for node {node_id}: {emission_value}")
            continue

    return total_emission


# Calculate global maximum mass flow
total_annual_emission = calculate_total_annual_emission(network_emission_flux)
seconds_per_year = 365.25 * 24 * 3600
global_max_massflow_kg_s = total_annual_emission / seconds_per_year

print(f"📊 Total annual emission: {total_annual_emission:,.0f} kg/year")
print(f"📊 Global max mass flow: {global_max_massflow_kg_s:.3f} kg/s")


def get_pipeline_directions_and_flows(pipeline_name, network_nodes, network_pipeline, network_emission_flux,
                                      global_max_massflow_kg_s):
    """
    Get all possible directions for a pipeline and calculate mass flows for each direction
    UPDATED: Fixed to handle pipeline 13_14 and 1_11 before transport checks
    """
    try:
        parts = pipeline_name.split('_')
        if len(parts) != 2:
            print(f"   ❌ Invalid pipeline name format: {pipeline_name}")
            return []

        node1, node2 = int(parts[0]), int(parts[1])

        # SPECIAL CASE: Handle pipeline 13_14 first (before transport checks)
        if pipeline_name == '13_14':
            print(f"   🔧 Special handling for pipeline 13_14 (offshore, fixed distance)")
            # Pipeline 13_14: Use specified parameters
            # Only analyze direction 13->14 as specified
            min_flow_t_h = 12.72  # Minimum 12.75 t/h
            max_flow_t_h = 670.0  # Maximum 670 t/h
            min_flow_kg_s = min_flow_t_h * 1000 / 3600
            max_flow_kg_s = max_flow_t_h * 1000 / 3600
            fixed_distance_km = 24.92  # Fixed distance as specified

            directions = [{
                'direction': f"{node1}_to_{node2}",
                'from_node': node1,
                'to_node': node2,
                'massflow_min_kg_per_s': round(min_flow_kg_s, 3),
                'massflow_max_kg_per_s': round(max_flow_kg_s, 3),
                'source_emission_t_h': min_flow_t_h,
                'distance_km': fixed_distance_km,  # Use fixed distance
                'terrain': 'Offshore'  # Offshore terrain as specified
            }]

            print(
                f"   ✅ Created direction 13→14: {min_flow_t_h} - {max_flow_t_h} t/h, {fixed_distance_km} km (Offshore)")
            return directions

        # SPECIAL CASE: Handle pipeline 1_11 (after 13_14 handling)
        if pipeline_name == '1_11':
            print(f"   🔧 Special handling for pipeline 1_11 (onshore, fixed distance)")
            # Pipeline 1_11: Use specified parameters
            # Only analyze direction 1->11 as specified
            min_flow_t_h = 38.64  # Minimum 38.64 t/h
            max_flow_t_h = 670.0  # Maximum 670 t/h (same as other pipelines)
            min_flow_kg_s = min_flow_t_h * 1000 / 3600
            max_flow_kg_s = max_flow_t_h * 1000 / 3600
            fixed_distance_km = 75.66  # Fixed distance as specified

            directions = [{
                'direction': f"{node1}_to_{node2}",
                'from_node': node1,
                'to_node': node2,
                'massflow_min_kg_per_s': round(min_flow_kg_s, 3),
                'massflow_max_kg_per_s': round(max_flow_kg_s, 3),
                'source_emission_t_h': min_flow_t_h,
                'distance_km': fixed_distance_km,  # Use fixed distance
                'terrain': 'Onshore'  # Onshore terrain
            }]

            print(
                f"   ✅ Created direction 1→11: {min_flow_t_h} - {max_flow_t_h} t/h, {fixed_distance_km} km (Onshore)")
            return directions

        # Check transport possibilities using network_pipeline (now contains distances, >0 means connected)
        can_transport_1_to_2 = False
        can_transport_2_to_1 = False
        distance_1_to_2 = 0
        distance_2_to_1 = 0

        if (node1 in network_pipeline.columns and node2 in network_pipeline.index):
            distance_1_to_2 = network_pipeline.loc[node2, node1]
            can_transport_1_to_2 = not pd.isna(distance_1_to_2) and distance_1_to_2 > 0

        if (node2 in network_pipeline.columns and node1 in network_pipeline.index):
            distance_2_to_1 = network_pipeline.loc[node1, node2]
            can_transport_2_to_1 = not pd.isna(distance_2_to_1) and distance_2_to_1 > 0

        if not (can_transport_1_to_2 or can_transport_2_to_1):
            print(f"   ❌ No transport possible for pipeline {pipeline_name}")
            return []

        directions = []

        # Handle pipeline 5_6 (normal case using network data)
        if pipeline_name == '5_6':
            # Pipeline 5_6: Use existing parameters
            max_flow_t_h = 670.0  # Maximum for both directions
            max_flow_kg_s = max_flow_t_h * 1000 / 3600  # Convert to kg/s

            if can_transport_1_to_2:
                # Direction 5->6: minimum 99.00 t/h
                min_flow_t_h = 99.00
                min_flow_kg_s = min_flow_t_h * 1000 / 3600

                directions.append({
                    'direction': f"{node1}_to_{node2}",
                    'from_node': node1,
                    'to_node': node2,
                    'massflow_min_kg_per_s': round(min_flow_kg_s, 3),
                    'massflow_max_kg_per_s': round(max_flow_kg_s, 3),
                    'source_emission_t_h': min_flow_t_h,
                    'distance_km': distance_1_to_2,
                    'terrain': 'Onshore'
                })

            if can_transport_2_to_1:
                # Direction 6->5: minimum 104.88 t/h
                min_flow_t_h = 104.88
                min_flow_kg_s = min_flow_t_h * 1000 / 3600

                directions.append({
                    'direction': f"{node2}_to_{node1}",
                    'from_node': node2,
                    'to_node': node1,
                    'massflow_min_kg_per_s': round(min_flow_kg_s, 3),
                    'massflow_max_kg_per_s': round(max_flow_kg_s, 3),
                    'source_emission_t_h': min_flow_t_h,
                    'distance_km': distance_2_to_1,
                    'terrain': 'Onshore'
                })

        return directions

    except Exception as e:
        print(f"   ❌ Error analyzing pipeline {pipeline_name}: {e}")
        return [{'direction': pipeline_name, 'from_node': None, 'to_node': None,
                 'massflow_min_kg_per_s': 1.000, 'massflow_max_kg_per_s': global_max_massflow_kg_s,
                 'source_emission_t_h': 0, 'distance_km': 0, 'terrain': 'Onshore'}]


def get_pipeline_length(pipeline_name, network_pipeline):
    """
    Get pipeline length from network pipeline matrix (which now contains distances)
    UPDATED: Handle special cases for pipeline 13_14 and 1_11 with fixed distances

    Args:
        pipeline_name: String like "5_6", "13_14", or "1_11"
        network_pipeline: Pipeline matrix DataFrame with distances (value > 0 means connected)

    Returns:
        float: Length in km, or None if not found
    """
    try:
        # Special case for pipeline 13_14
        if pipeline_name == '13_14':
            return 24.92  # Fixed distance as specified

        # Special case for pipeline 1_11
        if pipeline_name == '1_11':
            return 75.66  # Fixed distance as specified

        parts = pipeline_name.split('_')
        if len(parts) != 2:
            return None

        node1, node2 = int(parts[0]), int(parts[1])

        # Check if nodes exist in the pipeline matrix
        distance = None

        if node1 in network_pipeline.columns and node2 in network_pipeline.index:
            distance = network_pipeline.loc[node2, node1]
            if not pd.isna(distance) and distance > 0:
                return round(float(distance), 3)

        if node2 in network_pipeline.columns and node1 in network_pipeline.index:
            distance = network_pipeline.loc[node1, node2]
            if not pd.isna(distance) and distance > 0:
                return round(float(distance), 3)

        return None

    except Exception as e:
        print(f"   ❌ Error getting pipeline length for {pipeline_name}: {e}")
        return None


def get_actual_model_data_points(model_class, base_options, massflow_points_kg_s, model_name):
    """
    Calculate actual CAPEX data points by running the model at specific mass flow rates
    UPDATED: Now uses 20 data points instead of 8

    Args:
        model_class: Either OriginalModel or EnhancedModel class
        base_options: Base options dictionary for the model
        massflow_points_kg_s: Array of mass flow rates in kg/s to evaluate
        model_name: String name for debugging ("Original" or "Enhanced")

    Returns:
        tuple: (massflow_points_t_h, actual_capex_values)
    """
    print(f"   🔍 Calculating actual data points for {model_name} model...")

    massflow_points_t_h = massflow_points_kg_s / 1000 * 3600  # Convert to t/h
    actual_capex_values = []

    # First, get the gamma values from the original full-range calculation for reference
    model_instance_ref = model_class("CO2_Pipeline")
    results_ref = model_instance_ref.calculate_indicators(base_options.copy())
    gamma1_ref = results_ref['financial_indicators']['gamma1']
    gamma2_ref = results_ref['financial_indicators']['gamma2']

    print(f"      Reference γ₁: {gamma1_ref:,.0f} EUR, γ₂: {gamma2_ref:.3f} EUR/(t/h)")

    for i, massflow_kg_s in enumerate(massflow_points_kg_s):
        try:
            # Create fresh model instance for each calculation
            model_instance = model_class("CO2_Pipeline")

            # Update options for this specific mass flow rate with a small range around the point
            massflow_range_kg_s = massflow_kg_s * 0.01  # 1% range around the point
            point_options = base_options.copy()
            point_options.update({
                "massflow_min_kg_per_s": max(0.1, massflow_kg_s - massflow_range_kg_s),
                "massflow_max_kg_per_s": massflow_kg_s + massflow_range_kg_s,
                "massflow_evaluation_points": 3  # Small number of points around target
            })

            # Calculate indicators
            results = model_instance.calculate_indicators(point_options)

            # Debug: Print available keys in results
            if i == 0:
                print(f"      Debug - Available result keys: {list(results.keys())}")
                if 'financial_indicators' in results:
                    print(f"      Debug - Financial indicator keys: {list(results['financial_indicators'].keys())}")

            # Extract actual CAPEX using the linear model (this is what the models actually output)
            gamma1 = results['financial_indicators']['gamma1']
            gamma2 = results['financial_indicators']['gamma2']
            massflow_t_h = massflow_kg_s / 1000 * 3600

            # The "actual" cost according to the model's internal calculation
            capex = gamma1 + gamma2 * massflow_t_h
            actual_capex_values.append(capex)

            if i == 0:  # Debug info for first point
                print(f"      First point - Flow: {massflow_kg_s:.3f} kg/s ({massflow_t_h:.1f} t/h)")
                print(f"      γ₁: {gamma1:,.0f}, γ₂: {gamma2:.3f}, CAPEX: {capex:,.0f} EUR")

        except Exception as e:
            print(f"      ⚠️  Error at mass flow {massflow_kg_s:.3f} kg/s: {e}")
            # Use reference linear approximation as fallback
            massflow_t_h = massflow_kg_s / 1000 * 3600
            capex_fallback = gamma1_ref + gamma2_ref * massflow_t_h
            actual_capex_values.append(capex_fallback)

    print(f"      ✅ Calculated {len(actual_capex_values)} actual data points")
    print(f"      CAPEX range: {min(actual_capex_values):,.0f} - {max(actual_capex_values):,.0f} EUR")
    return massflow_points_t_h, np.array(actual_capex_values)


def compare_pipeline_costs(pipeline_name, direction_config, length_km, soil_data, anthro_data, morpho_data,
                           intersection_data):
    """Compare costs between original and enhanced models for a single pipeline direction with real mass flow data"""

    direction = direction_config['direction']
    from_node = direction_config['from_node']
    to_node = direction_config['to_node']
    massflow_min_kg_s = direction_config['massflow_min_kg_per_s']
    massflow_max_kg_s = direction_config['massflow_max_kg_per_s']
    terrain = direction_config.get('terrain', 'Onshore')  # Get terrain from direction config

    print(f"\n{'=' * 80}")
    print(f"COST COMPARISON FOR PIPELINE {pipeline_name} - DIRECTION {direction}")
    print(f"{'=' * 80}")
    print(f"Length: {length_km:.2f} km")
    print(f"Terrain: {terrain}")  # Display terrain
    print(f"From Node: {from_node} → To Node: {to_node}")
    print(f"Mass flow range: {massflow_min_kg_s:.3f} - {massflow_max_kg_s:.3f} kg/s")

    # Create evaluation range - UPDATED: Use 20 points for better resolution
    num_points = 20  # UPDATED: Increased from 10 to 20
    massflow_range_kg_s = np.linspace(massflow_min_kg_s, massflow_max_kg_s, num_points)

    # Common options for both models
    base_options = {
        "length_km": length_km,
        "currency_out": "EUR",
        "financial_year_out": 2024,
        "discount_rate": 0.1,
        "massflow_min_kg_per_s": massflow_min_kg_s,
        "massflow_max_kg_per_s": massflow_max_kg_s,
        "massflow_evaluation_points": num_points,
        "terrain": terrain,  # Use terrain from direction config
        "timeframe": "mid-term",
        "electricity_price_eur_per_mw": avg_electricity_price_eur_mwh
    }

    # 1. Calculate costs with ORIGINAL model
    print("\n1. Calculating costs with ORIGINAL model...")
    try:
        model_original = OriginalModel("CO2_Pipeline")
        results_original = model_original.calculate_indicators(base_options.copy())
        print(f"   Original γ₁: {results_original['financial_indicators']['gamma1']:,.0f} EUR")
        print(f"   Original γ₂: {results_original['financial_indicators']['gamma2']:,.3f} EUR/(t/h)")
    except Exception as e:
        print(f"   ❌ Error with original model calculation: {e}")
        raise

    # 2. Calculate costs with ENHANCED model
    print("\n2. Calculating costs with ENHANCED model...")

    # Create enhanced model instance
    try:
        model_enhanced = EnhancedModel("CO2_Pipeline")
    except Exception as e:
        print(f"   ❌ Error creating enhanced model: {e}")
        raise

    enhanced_options = base_options.copy()

    # Add geographical data to options if available
    if pipeline_name in intersection_data:
        print(f"\n3. Adding geographical data for pipeline {pipeline_name}")

        raw_grids = intersection_data[pipeline_name]['intersected_grids']
        raw_proportions = intersection_data[pipeline_name]['intersected_proportions']

        # Convert grid IDs and clean data
        try:
            intersected_grids = []
            intersected_proportions = []

            for grid, prop in zip(raw_grids, raw_proportions):
                if pd.notna(grid) and pd.notna(prop):
                    try:
                        # Try converting to int
                        grid_clean = int(grid)
                    except (ValueError, TypeError):
                        # Keep original format
                        grid_clean = grid

                    intersected_grids.append(grid_clean)
                    intersected_proportions.append(float(prop))

            print(f"   ✅ Cleaned data: {len(intersected_grids)} grids")

            # Debug: Print first few grids and their data availability
            print(f"   🔍 Debug: Sample intersected grids: {intersected_grids[:3]}")

            # Check if grids exist in geographical data
            soil_grid_ids = set(soil_data['GRID_OID'].tolist()) if 'GRID_OID' in soil_data.columns else set()
            morpho_grid_ids = set(morpho_data['GRID_OID'].tolist()) if 'GRID_OID' in morpho_data.columns else set()
            anthro_grid_ids = set(anthro_data['GRID_OID'].tolist()) if 'GRID_OID' in anthro_data.columns else set()

            # Also try alternative column names
            if 'grid_id' in soil_data.columns:
                soil_grid_ids.update(soil_data['grid_id'].tolist())
            if 'grid_id' in morpho_data.columns:
                morpho_grid_ids.update(morpho_data['grid_id'].tolist())
            if 'grid_id' in anthro_data.columns:
                anthro_grid_ids.update(anthro_data['grid_id'].tolist())

            intersected_set = set(intersected_grids)

            soil_matches = len(intersected_set.intersection(soil_grid_ids))
            morpho_matches = len(intersected_set.intersection(morpho_grid_ids))
            anthro_matches = len(intersected_set.intersection(anthro_grid_ids))

            print(f"   🔍 Debug: Grid matches - Soil:{soil_matches}, Morpho:{morpho_matches}, Anthro:{anthro_matches}")

            if soil_matches > 0 and morpho_matches > 0 and anthro_matches > 0:
                print(f"   ✅ Grid matches found - ready for geographical analysis")
            else:
                print(
                    f"   ⚠️  Limited grid matches: Soil:{soil_matches}, Morpho:{morpho_matches}, Anthro:{anthro_matches}")

        except Exception as e:
            print(f"   ❌ Error processing grid data: {e}")
            intersected_grids = []
            intersected_proportions = []

        enhanced_options.update({
            "morpho_data": morpho_data,
            "soil_data": soil_data,
            "anthro_data": anthro_data,
            "intersected_grids": intersected_grids,
            "intersected_proportions": intersected_proportions
        })

    else:
        print(f"\n   ⚠️  No geographical data available for pipeline {pipeline_name}")
        # Provide empty geographical data
        enhanced_options.update({
            "morpho_data": pd.DataFrame(),
            "soil_data": pd.DataFrame(),
            "anthro_data": pd.DataFrame(),
            "intersected_grids": [],
            "intersected_proportions": []
        })

    try:
        results_enhanced = model_enhanced.calculate_indicators(enhanced_options)

        # Get geographical factors
        geo_factors = results_enhanced.get('geo_factors', pd.DataFrame())
        if hasattr(model_enhanced, 'geo_factors'):
            geo_factors = model_enhanced.geo_factors

        print(f"   Enhanced γ₁: {results_enhanced['financial_indicators']['gamma1']:,.0f} EUR")
        print(f"   Enhanced γ₂: {results_enhanced['financial_indicators']['gamma2']:,.3f} EUR/(t/h)")

        # Check if geographical factors were applied
        if not geo_factors.empty and 'incremental_geo_factor' in geo_factors.columns:
            factors = geo_factors['incremental_geo_factor']
            print(f"   🔍 Debug: Geo factors range: {factors.min():.6f} to {factors.max():.6f}")
            print(f"   🔍 Debug: Geo factors std: {factors.std():.6f}")

            if factors.std() > 1e-6:
                print(f"   ✅ Geographical factors applied (factor range: {factors.min():.3f} to {factors.max():.3f})")
            else:
                print(f"   ⚠️  Geographical factors constant ({factors.mean():.3f}) - limited impact")
        else:
            print(f"   ❌ No geographical factors found in results")

    except Exception as e:
        print(f"   ❌ Error with enhanced calculation: {e}")
        import traceback
        traceback.print_exc()
        raise

    # 4. Calculate actual data points for both models - UPDATED: Use 20 data points within specific range
    print(f"\n4. Calculating actual data points for model comparison...")

    # UPDATED: Use specific mass flow range for this direction
    massflow_min_t_h = massflow_min_kg_s * 3600 / 1000  # Convert back to t/h for display
    massflow_max_t_h = massflow_max_kg_s * 3600 / 1000  # Convert back to t/h for display

    num_data_points = 20  # UPDATED: Use 20 data points
    massflow_data_points_kg_s = np.linspace(massflow_min_kg_s, massflow_max_kg_s, num_data_points)

    print(
        f"   Using mass flow range: {massflow_min_t_h:.2f} - {massflow_max_t_h:.2f} t/h with {num_data_points} points")

    # Get actual data points from both models
    original_massflow_t_h, original_actual_capex = get_actual_model_data_points(
        OriginalModel, base_options, massflow_data_points_kg_s, "Original")

    enhanced_massflow_t_h, enhanced_actual_capex = get_actual_model_data_points(
        EnhancedModel, enhanced_options, massflow_data_points_kg_s, "Enhanced")

    # Compare results
    print(f"\n   📊 RESULTS COMPARISON:")

    gamma1_diff = results_enhanced['financial_indicators']['gamma1'] - results_original['financial_indicators'][
        'gamma1']
    gamma2_diff = results_enhanced['financial_indicators']['gamma2'] - results_original['financial_indicators'][
        'gamma2']

    if abs(gamma1_diff) < 1 and abs(gamma2_diff) < 0.01:
        print(f"   🔄 Models produce identical results (no geographical impact)")
    else:
        gamma1_rel_diff = (gamma1_diff / results_original['financial_indicators']['gamma1']) * 100
        gamma2_rel_diff = (gamma2_diff / results_original['financial_indicators']['gamma2']) * 100
        print(f"   📈 Cost changes: Δγ₁={gamma1_diff:+,.0f} EUR ({gamma1_rel_diff:+.1f}%), "
              f"Δγ₂={gamma2_diff:+,.3f} EUR/(t/h) ({gamma2_rel_diff:+.1f}%)")

    # Store actual data points in results for plotting
    results_original['actual_data_points'] = {
        'massflow_t_h': original_massflow_t_h,
        'capex': original_actual_capex
    }
    results_enhanced['actual_data_points'] = {
        'massflow_t_h': enhanced_massflow_t_h,
        'capex': enhanced_actual_capex
    }

    return results_original, results_enhanced, geo_factors, direction_config


def plot_cost_comparison(pipeline_name, direction_config, results_original, results_enhanced, geo_factors, length_km):
    """Create comprehensive cost comparison plots with real mass flow data and actual vs fitted data points
    UPDATED: Uses specific mass flow ranges for all pipelines and shows terrain"""

    direction = direction_config['direction']
    from_node = direction_config['from_node']
    to_node = direction_config['to_node']
    massflow_min_kg_s = direction_config['massflow_min_kg_per_s']
    massflow_max_kg_s = direction_config['massflow_max_kg_per_s']
    terrain = direction_config.get('terrain', 'Onshore')

    # Set up the plotting style - UPDATED: Using Crameri navia colors
    plt.style.use('default')

    # Create figure with adjusted spacing
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, height_ratios=[1.2, 1, 1])

    # UPDATED: Use actual mass flow range for this direction instead of fixed range
    plot_massflow_min_t_h = massflow_min_kg_s * 3600 / 1000  # Convert to t/h
    plot_massflow_max_t_h = massflow_max_kg_s * 3600 / 1000  # Convert to t/h
    massflow_range_t_h = np.linspace(plot_massflow_min_t_h, plot_massflow_max_t_h,
                                     50)  # More points for smooth plotting

    # Calculate costs using linear models (fitted lines)
    costs_original_fit = (results_original['financial_indicators']['gamma1'] +
                          results_original['financial_indicators']['gamma2'] * massflow_range_t_h)
    costs_enhanced_fit = (results_enhanced['financial_indicators']['gamma1'] +
                          results_enhanced['financial_indicators']['gamma2'] * massflow_range_t_h)

    # Get actual data points
    original_actual_data = results_original.get('actual_data_points', {})
    enhanced_actual_data = results_enhanced.get('actual_data_points', {})

    # 1. Main comparison plot with actual data points AND fitted lines
    ax1 = fig.add_subplot(gs[0, :])

    # Plot fitted lines (linear models) - UPDATED: Using Crameri navia colors
    ax1.plot(massflow_range_t_h, costs_original_fit / 1e6, color=ORIGINAL_COLOR, linestyle='-', linewidth=3,
             label='Original Model (Linear Fit)', alpha=0.8)

    # Check if the results are different
    costs_identical = np.allclose(costs_original_fit, costs_enhanced_fit, rtol=1e-5)

    if costs_identical:
        ax1.plot(massflow_range_t_h, costs_enhanced_fit / 1e6, color=ENHANCED_COLOR, linestyle='--', linewidth=2,
                 label='Enhanced Model (Linear Fit - identical)', alpha=0.6)
        comparison_note = " (Models produce identical results)"
    else:
        ax1.plot(massflow_range_t_h, costs_enhanced_fit / 1e6, color=ENHANCED_COLOR, linestyle='-', linewidth=3,
                 label='Enhanced Model (Linear Fit)', alpha=0.8)

        cost_ratio = np.mean(costs_enhanced_fit) / np.mean(costs_original_fit)
        comparison_note = f" (Enhanced costs {cost_ratio:.1f}x original)"

    # Plot actual data points (dots) - UPDATED: Using Crameri navia colors
    if original_actual_data and 'massflow_t_h' in original_actual_data:
        ax1.scatter(original_actual_data['massflow_t_h'], original_actual_data['capex'] / 1e6,
                    c=ORIGINAL_COLOR, s=80, alpha=0.8, marker='o', edgecolors='black', linewidth=1.5,
                    label='Original Model (Actual Data Points)', zorder=5)

    if enhanced_actual_data and 'massflow_t_h' in enhanced_actual_data:
        if costs_identical:
            ax1.scatter(enhanced_actual_data['massflow_t_h'], enhanced_actual_data['capex'] / 1e6,
                        c=ENHANCED_COLOR, s=60, alpha=0.6, marker='s', edgecolors='black', linewidth=1,
                        label='Enhanced Model (Actual Data Points - identical)', zorder=4)
        else:
            ax1.scatter(enhanced_actual_data['massflow_t_h'], enhanced_actual_data['capex'] / 1e6,
                        c=ENHANCED_COLOR, s=80, alpha=0.8, marker='s', edgecolors='black', linewidth=1.5,
                        label='Enhanced Model (Actual Data Points)', zorder=5)

    ax1.set_xlabel('Mass Flow Rate (t/h)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('CAPEX (Million EUR)', fontweight='bold', fontsize=12)  # UPDATED: Removed "Total"

    # UPDATED: Display actual flow range used with terrain information
    title_text = f'Pipeline {pipeline_name} - {direction} - Cost Comparison{comparison_note}\n'
    title_text += f'Length: {length_km:.2f} km | Terrain: {terrain} | Flow Range: {plot_massflow_min_t_h:.2f}-{plot_massflow_max_t_h:.2f} t/h | '
    title_text += f'Direction: Node {from_node} → Node {to_node}'

    ax1.set_title(title_text, fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # Calculate R² for linear fit quality assessment if we have actual data
    if original_actual_data and 'massflow_t_h' in original_actual_data:
        # Calculate fitted values at actual data points for R² calculation
        original_actual_massflow = original_actual_data['massflow_t_h']
        original_fitted_at_actual = (results_original['financial_indicators']['gamma1'] +
                                     results_original['financial_indicators']['gamma2'] * original_actual_massflow)

        # Calculate R² for original model with better error handling
        actual_values = original_actual_data['capex']
        fitted_values = original_fitted_at_actual

        # Check if values are reasonable
        if np.any(np.abs(actual_values) < 1e-6) and np.any(fitted_values > 1e6):
            r2_original = float('nan')  # Invalid comparison
            r2_text = "Original R² = Invalid (data mismatch)"
        else:
            ss_res_orig = np.sum((actual_values - fitted_values) ** 2)
            ss_tot_orig = np.sum((actual_values - np.mean(actual_values)) ** 2)
            r2_original = 1 - (ss_res_orig / ss_tot_orig) if ss_tot_orig > 1e-10 else 1.0

            # Clamp R² to reasonable range for display
            if r2_original < -10:
                r2_text = f"Original R² = {r2_original:.1e} (very poor fit)"
            elif r2_original < 0:
                r2_text = f"Original R² = {r2_original:.3f} (poor fit)"
            else:
                r2_text = f"Original R² = {r2_original:.4f}"

        # Add R² text to plot - UPDATED: Using Crameri navia colors
        ax1.text(0.02, 0.98, r2_text, transform=ax1.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=ORIGINAL_COLOR, alpha=0.3),
                 verticalalignment='top', fontsize=10, fontweight='bold')

    if enhanced_actual_data and 'massflow_t_h' in enhanced_actual_data and not costs_identical:
        # Calculate R² for enhanced model
        enhanced_actual_massflow = enhanced_actual_data['massflow_t_h']
        enhanced_fitted_at_actual = (results_enhanced['financial_indicators']['gamma1'] +
                                     results_enhanced['financial_indicators']['gamma2'] * enhanced_actual_massflow)

        actual_values_enh = enhanced_actual_data['capex']
        fitted_values_enh = enhanced_fitted_at_actual

        # Check if values are reasonable
        if np.any(np.abs(actual_values_enh) < 1e-6) and np.any(fitted_values_enh > 1e6):
            r2_enhanced = float('nan')  # Invalid comparison
            r2_text_enh = "Enhanced R² = Invalid (data mismatch)"
        else:
            ss_res_enh = np.sum((actual_values_enh - fitted_values_enh) ** 2)
            ss_tot_enh = np.sum((actual_values_enh - np.mean(actual_values_enh)) ** 2)
            r2_enhanced = 1 - (ss_res_enh / ss_tot_enh) if ss_tot_enh > 1e-10 else 1.0

            # Clamp R² to reasonable range for display
            if r2_enhanced < -10:
                r2_text_enh = f"Enhanced R² = {r2_enhanced:.1e} (very poor fit)"
            elif r2_enhanced < 0:
                r2_text_enh = f"Enhanced R² = {r2_enhanced:.3f} (poor fit)"
            else:
                r2_text_enh = f"Enhanced R² = {r2_enhanced:.4f}"

        ax1.text(0.02, 0.90, r2_text_enh, transform=ax1.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=ENHANCED_COLOR, alpha=0.3),
                 verticalalignment='top', fontsize=10, fontweight='bold')

    # 2. Cost difference plot - UPDATED: Using Crameri navia colors
    ax2 = fig.add_subplot(gs[1, 0])
    cost_difference = costs_enhanced_fit - costs_original_fit

    if costs_identical:
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.7)
        ax2.text(0.5, 0.5, 'No cost difference\n(Models identical)',
                 transform=ax2.transAxes, ha='center', va='center', fontsize=12)
    else:
        ax2.plot(massflow_range_t_h, cost_difference / 1e6, color=DIFF_COLOR, linewidth=2)

    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Mass Flow Rate (t/h)', fontweight='bold')
    ax2.set_ylabel('Cost Difference (Million EUR)', fontweight='bold')
    ax2.set_title('Absolute Cost Difference\n(Enhanced - Original)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Relative cost difference plot - UPDATED: Using Crameri navia colors
    ax3 = fig.add_subplot(gs[1, 1])

    if costs_identical:
        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.7)
        ax3.text(0.5, 0.5, 'No relative difference\n(Models identical)',
                 transform=ax3.transAxes, ha='center', va='center', fontsize=12)
    else:
        relative_difference = (cost_difference / costs_original_fit) * 100
        ax3.plot(massflow_range_t_h, relative_difference, color=REL_DIFF_COLOR, linewidth=2)

    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Mass Flow Rate (t/h)', fontweight='bold')
    ax3.set_ylabel('Relative Difference (%)', fontweight='bold')
    ax3.set_title('Relative Cost Difference\n(Enhanced - Original)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Geographical factors plot - UPDATED: Using Crameri navia colors and fixed bracket text
    ax4 = fig.add_subplot(gs[2, 0])
    if not geo_factors.empty and 'incremental_geo_factor' in geo_factors.columns:
        factor_col = 'incremental_geo_factor'
        factors = geo_factors[factor_col]

        ax4.plot(geo_factors.index, factors, color=GEO_COLOR,
                 linewidth=2, marker='o', markersize=6)
        ax4.set_xlabel('Mass Flow Rate (t/h)', fontweight='bold')
        ax4.set_ylabel('Geographical Factor', fontweight='bold')

        # Add debug info to title - UPDATED: Fixed the bracket text
        factor_std = factors.std()
        if factor_std < 1e-6:
            debug_info = " (Constant factors)"
        else:
            debug_info = f" (Varies: σ={factor_std:.4f})"

        ax4.set_title(f'Geographical Cost Factors{debug_info}', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.0, color='red', linestyle='--', alpha=0.7,
                    label='No adjustment (factor = 0.0)')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No geographical\nfactor data available',
                 transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title('Geographical Cost Factors', fontweight='bold')

    # 5. Model parameters comparison table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Create comparison table data
    comparison_data = [
        ['Parameter', 'Original Model', 'Enhanced Model', 'Difference'],
        ['γ₁ (EUR)', f"{results_original['financial_indicators']['gamma1']:,.0f}",
         f"{results_enhanced['financial_indicators']['gamma1']:,.0f}",
         f"{results_enhanced['financial_indicators']['gamma1'] - results_original['financial_indicators']['gamma1']:+,.0f}"],
        ['γ₂ (EUR/(t/h))', f"{results_original['financial_indicators']['gamma2']:,.3f}",
         f"{results_enhanced['financial_indicators']['gamma2']:,.3f}",
         f"{results_enhanced['financial_indicators']['gamma2'] - results_original['financial_indicators']['gamma2']:+,.3f}"],
        ['OPEX Variable (EUR/t)', f"{results_original['financial_indicators']['opex_variable']:,.3f}",
         f"{results_enhanced['financial_indicators']['opex_variable']:,.3f}",
         f"{results_enhanced['financial_indicators']['opex_variable'] - results_original['financial_indicators']['opex_variable']:+,.3f}"],
        ['OPEX Fixed (%)', f"{results_original['financial_indicators']['opex_fixed']:.3f}",
         f"{results_enhanced['financial_indicators']['opex_fixed']:.3f}",
         f"{results_enhanced['financial_indicators']['opex_fixed'] - results_original['financial_indicators']['opex_fixed']:+.3f}"],
        ['Levelized Cost (EUR/t)', f"{results_original['financial_indicators']['levelized_cost']:,.3f}",
         f"{results_enhanced['financial_indicators']['levelized_cost']:,.3f}",
         f"{results_enhanced['financial_indicators']['levelized_cost'] - results_original['financial_indicators']['levelized_cost']:+,.3f}"]
    ]

    # Create table with proper positioning - UPDATED: Even more horizontal stretch to prevent text cutoff
    table = ax5.table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                      cellLoc='center', loc='center',
                      bbox=[-0.1, 0.02, 1.2, 0.85])  # UPDATED: Extended width to 1.2 and shifted further left to -0.1
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # UPDATED: Increased font size from 9 to 10
    table.scale(1, 2.2)  # UPDATED: Increased row height from 1.8 to 2.2

    # Style the table
    for i in range(len(comparison_data)):
        for j in range(len(comparison_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')

    # Add title with emission info and terrain - UPDATED: Use the direct t/h value from direction config
    emission_t_h = direction_config.get('source_emission_t_h', 0)
    emission_info = f"Source emission: {emission_t_h:.2f} t/h | Terrain: {terrain}"

    ax5.set_title(f'Model Parameters Comparison\n{emission_info}',
                  fontweight='bold', y=0.95, fontsize=11)

    plt.suptitle(f'CO2 Pipeline Cost Analysis - {pipeline_name} ({direction})',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save the plot
    output_filename = f'pipeline_{pipeline_name}_{direction}_cost_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"   💾 Saved plot as: {output_filename}")

    # Show the plot
    plt.show()

    return fig


def print_summary_analysis(pipeline_name, direction_config, results_original, results_enhanced, geo_factors, length_km):
    """Print detailed summary analysis with real mass flow data"""

    direction = direction_config['direction']
    from_node = direction_config['from_node']
    to_node = direction_config['to_node']
    massflow_min_kg_s = direction_config['massflow_min_kg_per_s']
    massflow_max_kg_s = direction_config['massflow_max_kg_per_s']
    terrain = direction_config.get('terrain', 'Onshore')

    print(f"\n{'=' * 80}")
    print(f"DETAILED ANALYSIS SUMMARY - PIPELINE {pipeline_name} - DIRECTION {direction}")
    print(f"{'=' * 80}")

    # UPDATED: Display actual mass flow ranges used
    massflow_min_t_h = massflow_min_kg_s * 3600 / 1000
    massflow_max_t_h = massflow_max_kg_s * 3600 / 1000

    # Basic information
    print(f"Pipeline length: {length_km:.3f} km")
    print(f"Terrain: {terrain}")  # Display terrain
    print(f"Mass flow range: {massflow_min_t_h:.2f} - {massflow_max_t_h:.2f} t/h")
    print(f"Pipeline direction: Node {from_node} → Node {to_node}")
    print(f"Source emission: {direction_config.get('source_emission_t_h', 0):.2f} t/h")
    print(f"Data points calculated: 20 (updated from 8)")

    # Cost impact analysis
    gamma1_diff = results_enhanced['financial_indicators']['gamma1'] - results_original['financial_indicators'][
        'gamma1']
    gamma2_diff = results_enhanced['financial_indicators']['gamma2'] - results_original['financial_indicators'][
        'gamma2']

    # Check if results are identical
    if abs(gamma1_diff) < 1e-3 and abs(gamma2_diff) < 1e-6:
        print(f"\n🔄 MODELS PRODUCE IDENTICAL RESULTS")
        print(f"This indicates no geographical factors were applied")
        return

    gamma1_rel_diff = (gamma1_diff / results_original['financial_indicators']['gamma1']) * 100
    gamma2_rel_diff = (gamma2_diff / results_original['financial_indicators']['gamma2']) * 100

    print(f"\nCost Parameter Changes:")
    print(f"  γ₁ change: {gamma1_diff:+,.0f} EUR ({gamma1_rel_diff:+.2f}%)")
    print(f"  γ₂ change: {gamma2_diff:+,.3f} EUR/(t/h) ({gamma2_rel_diff:+.2f}%)")

    # Linear fit quality analysis
    original_actual_data = results_original.get('actual_data_points', {})
    enhanced_actual_data = results_enhanced.get('actual_data_points', {})

    if original_actual_data and 'massflow_t_h' in original_actual_data:
        print(f"\nLinear Fit Quality Analysis (20 data points):")

        # Calculate fitted values for original model
        original_actual_massflow = original_actual_data['massflow_t_h']
        original_fitted_at_actual = (results_original['financial_indicators']['gamma1'] +
                                     results_original['financial_indicators']['gamma2'] * original_actual_massflow)

        actual_values = original_actual_data['capex']
        fitted_values = original_fitted_at_actual

        # Check if we have a valid comparison
        if np.any(np.abs(actual_values) < 1e-6) and np.any(fitted_values > 1e6):
            print(f"  ⚠️  Original model: Invalid R² calculation (data mismatch)")
            print(f"      Actual values range: {actual_values.min():.1f} - {actual_values.max():.1f}")
            print(f"      Fitted values range: {fitted_values.min():,.0f} - {fitted_values.max():,.0f}")
        else:
            ss_res_orig = np.sum((actual_values - fitted_values) ** 2)
            ss_tot_orig = np.sum((actual_values - np.mean(actual_values)) ** 2)
            r2_original = 1 - (ss_res_orig / ss_tot_orig) if ss_tot_orig > 1e-10 else 1.0

            print(f"  Original model R²: {r2_original:.4f}")

            if np.isnan(r2_original) or r2_original < -10:
                print(f"  → Invalid R² - likely data extraction issue")
            elif r2_original > 0.99:
                print(f"  → Excellent linear fit (model is essentially linear)")
            elif r2_original > 0.95:
                print(f"  → Good linear fit")
            elif r2_original > 0.90:
                print(f"  → Reasonable linear fit")
            elif r2_original > 0:
                print(f"  → Poor linear fit - model may be non-linear")
            else:
                print(f"  → Very poor fit - check data quality")

    if enhanced_actual_data and 'massflow_t_h' in enhanced_actual_data:
        enhanced_actual_massflow = enhanced_actual_data['massflow_t_h']
        enhanced_fitted_at_actual = (results_enhanced['financial_indicators']['gamma1'] +
                                     results_enhanced['financial_indicators']['gamma2'] * enhanced_actual_massflow)

        actual_values_enh = enhanced_actual_data['capex']
        fitted_values_enh = enhanced_fitted_at_actual

        # Check if we have a valid comparison
        if np.any(np.abs(actual_values_enh) < 1e-6) and np.any(fitted_values_enh > 1e6):
            print(f"  ⚠️  Enhanced model: Invalid R² calculation (data mismatch)")
            print(f"      Actual values range: {actual_values_enh.min():.1f} - {actual_values_enh.max():.1f}")
            print(f"      Fitted values range: {fitted_values_enh.min():,.0f} - {fitted_values_enh.max():,.0f}")
        else:
            ss_res_enh = np.sum((actual_values_enh - fitted_values_enh) ** 2)
            ss_tot_enh = np.sum((actual_values_enh - np.mean(actual_values_enh)) ** 2)
            r2_enhanced = 1 - (ss_res_enh / ss_tot_enh) if ss_tot_enh > 1e-10 else 1.0

            print(f"  Enhanced model R²: {r2_enhanced:.4f}")

            if np.isnan(r2_enhanced) or r2_enhanced < -10:
                print(f"  → Invalid R² - likely data extraction issue")
            elif r2_enhanced > 0.99:
                print(f"  → Excellent linear fit (model is essentially linear)")
            elif r2_enhanced > 0.95:
                print(f"  → Good linear fit")
            elif r2_enhanced > 0.90:
                print(f"  → Reasonable linear fit")
            elif r2_enhanced > 0:
                print(f"  → Poor linear fit - geographical factors may introduce non-linearity")
            else:
                print(f"  → Very poor fit - check data quality")

    # Geographical factor analysis
    if not geo_factors.empty and 'incremental_geo_factor' in geo_factors.columns:
        factor_col = 'incremental_geo_factor'
        factors = geo_factors[factor_col]
        avg_geo_factor = factors.mean()
        min_geo_factor = factors.min()
        max_geo_factor = factors.max()
        factor_std = factors.std()

        print(f"\nGeographical Factor Analysis:")
        print(f"  Average factor: {avg_geo_factor:.3f} ({avg_geo_factor * 100:+.1f}% cost change)")
        print(f"  Range: {min_geo_factor:.3f} - {max_geo_factor:.3f}")
        print(f"  Standard deviation: {factor_std:.6f}")

        if factor_std < 1e-6:
            print(f"  ⚠️  CONSTANT FACTORS - Pipeline categories not changing with mass flow")
        else:
            print(f"  ✅ VARYING FACTORS - Enhanced model working correctly")

        if avg_geo_factor > 0.05:
            print(f"  → Terrain increases costs by ~{avg_geo_factor * 100:.1f}% on average")
        elif avg_geo_factor < -0.05:
            print(f"  → Terrain decreases costs by ~{abs(avg_geo_factor) * 100:.1f}% on average")
        else:
            print(f"  → Terrain has minimal impact on costs")
    else:
        print(f"\n❌ No geographical factor data available")


# ============================================================================
# MAIN ANALYSIS EXECUTION
# ============================================================================

def run_cost_comparison_analysis():
    """Run the complete cost comparison analysis for pipelines 5_6, 13_14, and 1_11 with specific mass flow ranges and 20 data points"""

    print(f"\n{'=' * 80}")
    print("STARTING COST COMPARISON ANALYSIS - PIPELINES 5_6, 13_14, AND 1_11")
    print(f"Updated with specific mass flow ranges:")
    print(f"  Pipeline 5_6:")
    print(f"    Direction 5→6: 99.00 t/h - 670.0 t/h (Onshore)")
    print(f"    Direction 6→5: 104.88 t/h - 670.0 t/h (Onshore)")
    print(f"  Pipeline 13_14:")
    print(f"    Direction 13→14: 12.72 t/h - 670.0 t/h (Offshore, 24.92 km)")
    print(f"  Pipeline 1_11:")
    print(f"    Direction 1→11: 38.64 t/h - 670.0 t/h (Onshore, 75.66 km)")
    print(f"  Using 20 actual data points for better resolution")
    print(f"  Using Crameri navia color palette for consistency")
    print(f"  FIXED: Pipeline 13_14 and 1_11 special handling before transport checks")
    print(f"{'=' * 80}")

    # Process all pipelines
    for pipeline_name in pipeline_names:
        print(f"\n{'=' * 80}")
        print(f"PROCESSING PIPELINE {pipeline_name}")
        print(f"{'=' * 80}")

        # Get all possible directions and their mass flows
        directions = get_pipeline_directions_and_flows(pipeline_name, network_nodes, network_pipeline,
                                                       network_emission_flux, global_max_massflow_kg_s)

        if not directions:
            print(f"Skipping pipeline {pipeline_name} - no valid directions found")
            continue

        # Process each direction
        for direction_config in directions:
            try:
                print(f"\n   Processing direction: {direction_config['direction']}")

                # Display the mass flow range and terrain for this direction
                min_t_h = direction_config['massflow_min_kg_per_s'] * 3600 / 1000
                max_t_h = direction_config['massflow_max_kg_per_s'] * 3600 / 1000
                terrain = direction_config.get('terrain', 'Onshore')
                print(f"   Mass flow range: {min_t_h:.2f} - {max_t_h:.2f} t/h")
                print(f"   Terrain: {terrain}")

                # Get length from the direction config
                length_km = direction_config.get('distance_km', 0)

                if length_km is None or length_km <= 0:
                    print(f"   ❌ No valid distance found for direction {direction_config['direction']}")
                    continue

                print(f"   Distance: {length_km:.2f} km")

                # Run cost comparison (now with 20 data points in specific ranges)
                results_original, results_enhanced, geo_factors, direction_config = compare_pipeline_costs(
                    pipeline_name, direction_config, length_km, soil_data, anthro_data, morpho_data, intersection_data)

                # Create plots (now uses actual mass flow ranges and shows terrain)
                plot_cost_comparison(pipeline_name, direction_config, results_original, results_enhanced,
                                     geo_factors, length_km)

                # Print summary analysis (now includes updated mass flow info and terrain)
                print_summary_analysis(pipeline_name, direction_config, results_original, results_enhanced,
                                       geo_factors, length_km)

            except Exception as e:
                print(f"❌ Error processing direction {direction_config['direction']} for pipeline {pipeline_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE - PIPELINES 5_6, 13_14, AND 1_11 WITH UPDATED PARAMETERS")
    print(f"{'=' * 80}")
    print("Key updates made:")
    print("  ✅ FIXED: Pipeline 13_14 and 1_11 special handling moved before transport checks")
    print("  ✅ Added pipeline 1_11 analysis")
    print("  ✅ Pipeline 5_6 parameters:")
    print("      • Direction 5→6 minimum: 99.00 t/h")
    print("      • Direction 6→5 minimum: 104.88 t/h")
    print("      • Maximum: 670.0 t/h for both directions")
    print("      • Terrain: Onshore")
    print("  ✅ Pipeline 13_14 parameters:")
    print("      • Direction 13→14 only")
    print("      • Minimum: 12.72 t/h")
    print("      • Maximum: 670.0 t/h")
    print("      • Distance: 24.92 km (fixed)")
    print("      • Terrain: Offshore")
    print("  ✅ Pipeline 1_11 parameters:")
    print("      • Direction 1→11 only")
    print("      • Minimum: 38.64 t/h")
    print("      • Maximum: 670.0 t/h")
    print("      • Distance: 75.66 km (fixed)")
    print("      • Terrain: Onshore")
    print("  ✅ Increased data points from 8 to 20")
    print("  ✅ Updated plot ranges to use actual mass flow ranges")
    print("  ✅ Applied Crameri navia color palette for consistency")
    print("  ✅ Added terrain information to plots and summaries")
    print("Now all three pipelines (5_6, 13_14, and 1_11) should generate plots correctly!")


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    run_cost_comparison_analysis()