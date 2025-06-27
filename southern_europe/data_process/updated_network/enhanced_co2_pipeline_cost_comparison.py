import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns


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
network_distance = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0, sheet_name='distances')
network_pipeline = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0,
                                 sheet_name='pipeline_transport')

# Load electricity data
electricity_price = pd.read_csv(path_files_electricity/"electricity_prices_hourly_2024.csv") # electricity price

# Load intersection data
intersection_file = path_files_node_flux / "route_grid_intersections.xlsx"
pipeline_names = ['5_6', '5_4', '5_3', '2_1']
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

#----- Electricity price calculation -----#
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

#----- Emission calculation and mass flow determination -----#

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

    Args:
        pipeline_name: String like "2_1"
        network_nodes: DataFrame with node information
        network_pipeline: Binary matrix indicating transport possibilities (1 = transport from column to row)
        network_emission_flux: DataFrame with emission data
        global_max_massflow_kg_s: Global maximum mass flow rate for all pipelines

    Returns:
        list: List of direction dictionaries with mass flow data
    """
    try:
        parts = pipeline_name.split('_')
        if len(parts) != 2:
            print(f"   ❌ Invalid pipeline name format: {pipeline_name}")
            return []

        node1, node2 = int(parts[0]), int(parts[1])

        # Check transport possibilities using network_pipeline
        can_transport_1_to_2 = False
        can_transport_2_to_1 = False

        if (node1 in network_pipeline.columns and node2 in network_pipeline.index):
            can_transport_1_to_2 = bool(network_pipeline.loc[node2, node1])

        if (node2 in network_pipeline.columns and node1 in network_pipeline.index):
            can_transport_2_to_1 = bool(network_pipeline.loc[node1, node2])

        if not (can_transport_1_to_2 or can_transport_2_to_1):
            print(f"   ❌ No transport possible for pipeline {pipeline_name}")
            return []

        # Get emissions for both nodes
        def get_node_emission(node_id):
            if node_id not in network_emission_flux.index:
                return 0

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

            # Convert to scalar and return
            try:
                return float(emission_value) if not pd.isna(emission_value) else 0
            except (ValueError, TypeError):
                return 0

        emission_node1 = get_node_emission(node1)
        emission_node2 = get_node_emission(node2)

        # Convert to kg/s
        emission_node1_kg_s = float(emission_node1) / seconds_per_year
        emission_node2_kg_s = float(emission_node2) / seconds_per_year

        # Create direction configurations with standardized max flow
        directions = []

        if can_transport_1_to_2:
            min_flow = max(emission_node1_kg_s, 0.100)  # Minimum based on source emission
            max_flow = global_max_massflow_kg_s  # Same for all pipelines

            directions.append({
                'direction': f"{node1}_to_{node2}",
                'from_node': node1,
                'to_node': node2,
                'massflow_min_kg_per_s': round(min_flow, 3),
                'massflow_max_kg_per_s': round(max_flow, 3),
                'source_emission_kg_year': emission_node1
            })

        if can_transport_2_to_1:
            min_flow = max(emission_node2_kg_s, 0.100)  # Minimum based on source emission
            max_flow = global_max_massflow_kg_s  # Same for all pipelines

            directions.append({
                'direction': f"{node2}_to_{node1}",
                'from_node': node2,
                'to_node': node1,
                'massflow_min_kg_per_s': round(min_flow, 3),
                'massflow_max_kg_per_s': round(max_flow, 3),
                'source_emission_kg_year': emission_node2
            })

        return directions

    except Exception as e:
        print(f"   ❌ Error analyzing pipeline {pipeline_name}: {e}")
        return [{'direction': pipeline_name, 'from_node': None, 'to_node': None,
                 'massflow_min_kg_per_s': 1.000, 'massflow_max_kg_per_s': global_max_massflow_kg_s,
                 'source_emission_kg_year': 0}]


def get_pipeline_length(pipeline_name, network_distance):
    """
    Get pipeline length from network distance matrix

    Args:
        pipeline_name: String like "2_1"
        network_distance: Distance matrix DataFrame

    Returns:
        float: Length in km, or None if not found
    """
    try:
        parts = pipeline_name.split('_')
        if len(parts) != 2:
            return None

        node1, node2 = int(parts[0]), int(parts[1])

        # Check if nodes exist in the distance matrix
        if node1 not in network_distance.index or node2 not in network_distance.columns:
            if node1 not in network_distance.columns or node2 not in network_distance.index:
                return None

        # Try to get distance from the matrix
        distance = None

        if node1 in network_distance.index and node2 in network_distance.columns:
            distance = network_distance.loc[node1, node2]

        if (
                pd.isna(
                    distance) or distance == 0) and node2 in network_distance.index and node1 in network_distance.columns:
            # Try the reverse direction
            distance = network_distance.loc[node2, node1]

        if pd.isna(distance) or distance == 0:
            return None

        return round(float(distance), 3)

    except Exception as e:
        print(f"   ❌ Error getting pipeline length for {pipeline_name}: {e}")
        return None


def compare_pipeline_costs(pipeline_name, direction_config, length_km, soil_data, anthro_data, morpho_data,
                           intersection_data):
    """Compare costs between original and enhanced models for a single pipeline direction with real mass flow data"""

    direction = direction_config['direction']
    from_node = direction_config['from_node']
    to_node = direction_config['to_node']
    massflow_min_kg_s = direction_config['massflow_min_kg_per_s']
    massflow_max_kg_s = direction_config['massflow_max_kg_per_s']

    print(f"\n{'=' * 80}")
    print(f"COST COMPARISON FOR PIPELINE {pipeline_name} - DIRECTION {direction}")
    print(f"{'=' * 80}")
    print(f"Length: {length_km:.3f} km")
    print(f"From Node: {from_node} → To Node: {to_node}")
    print(f"Mass flow range: {massflow_min_kg_s:.3f} - {massflow_max_kg_s:.3f} kg/s")

    # Create evaluation range
    num_points = 10
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
        "terrain": "Onshore",
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

    return results_original, results_enhanced, geo_factors, direction_config


def plot_cost_comparison(pipeline_name, direction_config, results_original, results_enhanced, geo_factors, length_km):
    """Create comprehensive cost comparison plots with real mass flow data"""

    direction = direction_config['direction']
    from_node = direction_config['from_node']
    to_node = direction_config['to_node']
    massflow_min_kg_s = direction_config['massflow_min_kg_per_s']
    massflow_max_kg_s = direction_config['massflow_max_kg_per_s']

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with adjusted spacing
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, height_ratios=[1.2, 1, 1])

    # Define mass flow range for plotting
    massflow_range_kg_s = np.linspace(massflow_min_kg_s, massflow_max_kg_s, 50)  # More points for smooth plotting
    massflow_range_t_h = massflow_range_kg_s / 1000 * 3600  # Convert to t/h

    # Calculate costs using linear models
    costs_original = (results_original['financial_indicators']['gamma1'] +
                      results_original['financial_indicators']['gamma2'] * massflow_range_t_h)
    costs_enhanced = (results_enhanced['financial_indicators']['gamma1'] +
                      results_enhanced['financial_indicators']['gamma2'] * massflow_range_t_h)

    # 1. Main comparison plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(massflow_range_t_h, costs_original / 1e6, 'b-', linewidth=3,
             label='Original Model', alpha=0.8)

    # Check if the results are different
    costs_identical = np.allclose(costs_original, costs_enhanced, rtol=1e-5)

    if costs_identical:
        ax1.plot(massflow_range_t_h, costs_enhanced / 1e6, 'r--', linewidth=2,
                 label='Enhanced Model (identical)', alpha=0.6)
        comparison_note = " (Models produce identical results)"
    else:
        ax1.plot(massflow_range_t_h, costs_enhanced / 1e6, 'r-', linewidth=3,
                 label='Enhanced Model (with geo factors)', alpha=0.8)

        cost_ratio = np.mean(costs_enhanced) / np.mean(costs_original)
        comparison_note = f" (Enhanced costs {cost_ratio:.1f}x original)"

    ax1.set_xlabel('Mass Flow Rate (t/h)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Total CAPEX (Million EUR)', fontweight='bold', fontsize=12)

    title_text = f'Pipeline {pipeline_name} - {direction} - Cost Comparison{comparison_note}\n'
    title_text += f'Length: {length_km:.3f} km | Flow: {massflow_min_kg_s:.3f}-{massflow_max_kg_s:.3f} kg/s | '
    title_text += f'Direction: Node {from_node} → Node {to_node}'

    ax1.set_title(title_text, fontweight='bold', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Cost difference plot
    ax2 = fig.add_subplot(gs[1, 0])
    cost_difference = costs_enhanced - costs_original

    if costs_identical:
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.7)
        ax2.text(0.5, 0.5, 'No cost difference\n(Models identical)',
                 transform=ax2.transAxes, ha='center', va='center', fontsize=12)
    else:
        ax2.plot(massflow_range_t_h, cost_difference / 1e6, 'g-', linewidth=2)

    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Mass Flow Rate (t/h)', fontweight='bold')
    ax2.set_ylabel('Cost Difference (Million EUR)', fontweight='bold')
    ax2.set_title('Absolute Cost Difference\n(Enhanced - Original)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Relative cost difference plot
    ax3 = fig.add_subplot(gs[1, 1])

    if costs_identical:
        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.7)
        ax3.text(0.5, 0.5, 'No relative difference\n(Models identical)',
                 transform=ax3.transAxes, ha='center', va='center', fontsize=12)
    else:
        relative_difference = (cost_difference / costs_original) * 100
        ax3.plot(massflow_range_t_h, relative_difference, 'purple', linewidth=2)

    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Mass Flow Rate (t/h)', fontweight='bold')
    ax3.set_ylabel('Relative Difference (%)', fontweight='bold')
    ax3.set_title('Relative Cost Difference\n(Enhanced - Original)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Geographical factors plot
    ax4 = fig.add_subplot(gs[2, 0])
    if not geo_factors.empty and 'incremental_geo_factor' in geo_factors.columns:
        factor_col = 'incremental_geo_factor'
        factors = geo_factors[factor_col]

        ax4.plot(geo_factors.index, factors, 'orange',
                 linewidth=2, marker='o', markersize=6)
        ax4.set_xlabel('Mass Flow Rate (t/h)', fontweight='bold')
        ax4.set_ylabel('Geographical Factor', fontweight='bold')

        # Add debug info to title
        factor_std = factors.std()
        if factor_std < 1e-6:
            debug_info = " (⚠️ Constant factors)"
        else:
            debug_info = f" (✅ Varies: σ={factor_std:.4f})"

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

    # Create table with proper positioning
    table = ax5.table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                      cellLoc='center', loc='center', bbox=[0, 0.15, 1, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style the table
    for i in range(len(comparison_data)):
        for j in range(len(comparison_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')

    # Add title with emission info
    emission_info = f"Source emission: {direction_config.get('source_emission_kg_year', 0):,.0f} kg/year"
    ax5.set_title(f'Model Parameters Comparison\n{emission_info}', fontweight='bold', y=0.95, fontsize=12)

    plt.suptitle(f'CO2 Pipeline Cost Analysis - {pipeline_name} ({direction})',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save the plot
    output_filename = f'pipeline_{pipeline_name}_{direction}_cost_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"   💾 Saved plot as: {output_filename}")

    # Show the plot in PyCharm console
    plt.show()

    return fig


def print_summary_analysis(pipeline_name, direction_config, results_original, results_enhanced, geo_factors, length_km):
    """Print detailed summary analysis with real mass flow data"""

    direction = direction_config['direction']
    from_node = direction_config['from_node']
    to_node = direction_config['to_node']
    massflow_min_kg_s = direction_config['massflow_min_kg_per_s']
    massflow_max_kg_s = direction_config['massflow_max_kg_per_s']

    print(f"\n{'=' * 80}")
    print(f"DETAILED ANALYSIS SUMMARY - PIPELINE {pipeline_name} - DIRECTION {direction}")
    print(f"{'=' * 80}")

    # Basic information
    print(f"Pipeline length: {length_km:.3f} km")
    print(f"Mass flow range: {massflow_min_kg_s:.3f} - {massflow_max_kg_s:.3f} kg/s (based on real emissions)")
    print(f"Pipeline direction: Node {from_node} → Node {to_node}")
    print(f"Source emission: {direction_config.get('source_emission_kg_year', 0):,.0f} kg/year")

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
    """Run the complete cost comparison analysis for all pipelines with real mass flow data"""

    print(f"\n{'=' * 80}")
    print("STARTING COST COMPARISON ANALYSIS WITH REAL MASS FLOW DATA")
    print(f"{'=' * 80}")

    # Process each pipeline
    for pipeline_name in pipeline_names:
        print(f"\n{'=' * 80}")
        print(f"PROCESSING PIPELINE {pipeline_name}")
        print(f"{'=' * 80}")

        # Get pipeline length
        length_km = get_pipeline_length(pipeline_name, network_distance)

        if length_km is None:
            print(f"Skipping pipeline {pipeline_name} - no length data available")
            continue

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

                # Run cost comparison
                results_original, results_enhanced, geo_factors, direction_config = compare_pipeline_costs(
                    pipeline_name, direction_config, length_km, soil_data, anthro_data, morpho_data, intersection_data)

                # Create plots
                plot_cost_comparison(pipeline_name, direction_config, results_original, results_enhanced,
                                     geo_factors, length_km)

                # Print summary analysis
                print_summary_analysis(pipeline_name, direction_config, results_original, results_enhanced,
                                       geo_factors, length_km)

            except Exception as e:
                print(f"❌ Error processing direction {direction_config['direction']} for pipeline {pipeline_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print("Check the generated PNG files for detailed cost comparison plots.")
    print("Mass flow ranges are now based on real emission data from the nodes.")
    print("Bidirectional pipelines have been analyzed in both directions.")


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    run_cost_comparison_analysis()