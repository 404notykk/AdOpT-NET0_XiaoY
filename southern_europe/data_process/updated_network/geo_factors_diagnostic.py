import sys
import os
import pandas as pd
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from enhanced_co2_pipelines_cost_model import CO2_Pipeline_CostModel

print("=" * 80)
print("GEOGRAPHICAL FACTOR CALCULATION DIAGNOSTIC")
print("=" * 80)

# Load data
path_data_case_study = Path("../../northern_italy_data")
if not path_data_case_study.exists():
    path_data_case_study = Path("../../../northern_italy_data")

path_files_grids = path_data_case_study / "geographical_feature"

# Load geographical data
soil_data = pd.read_csv(path_files_grids / "soil_type_grids_italy.csv")
anthro_data = pd.read_csv(path_files_grids / "anthropisation_grids_italy.csv")
morpho_data = pd.read_csv(path_files_grids / "morphological_feature_grids_italy.csv")

# Load intersection data for pipeline 2_1
intersection_file = path_files_grids / "route_grid_intersections.xlsx"
df_2_1 = pd.read_excel(intersection_file, sheet_name='2_1')

# Handle metadata row if present
if len(df_2_1) > 0 and str(df_2_1.iloc[0]['grid_id']).startswith('Route:'):
    df_2_1 = df_2_1.iloc[1:].reset_index(drop=True)

intersected_grids = df_2_1['grid_id'].tolist()
intersected_proportions = df_2_1['proportion'].tolist()

print(f"Testing pipeline 2_1:")
print(f"Intersected grids: {intersected_grids}")
print(f"Proportions: {intersected_proportions}")

# Create enhanced model instance
model = CO2_Pipeline_CostModel("CO2_Pipeline")

# Test geographical factor calculation step by step
print(f"\n1. TESTING GEOGRAPHICAL FACTOR CALCULATION:")
print("-" * 50)

# Get cost factors
massflow_kg_per_s = 20.0  # Test with 20 kg/s
pipeline_category = model._get_pipeline_category_from_massflow(massflow_kg_per_s)
cost_factors = model._get_cost_factors(pipeline_category)

print(f"Test mass flow: {massflow_kg_per_s} kg/s")
print(f"Pipeline category (DN): {pipeline_category}")
print(f"Cost factors:")
for factor_name, factor_value in cost_factors.items():
    print(f"  {factor_name}: {factor_value}")

# Manual calculation to debug the issue
print(f"\n2. MANUAL STEP-BY-STEP CALCULATION:")
print("-" * 50)

total_weighted_factor = 0.0
total_weight = 0.0

for grid_id, intersection_prop in zip(intersected_grids, intersected_proportions):
    print(f"\nProcessing Grid {grid_id} (intersection proportion: {intersection_prop:.4f}):")

    # Get morphological proportions
    morpho_grid = morpho_data[morpho_data['GRID_OID'] == grid_id]
    if morpho_grid.empty:
        print(f"  ⚠️  No morphological data for grid {grid_id}")
        continue

    plain_prop = morpho_grid['PLAIN_M'].iloc[0]
    hill_prop = morpho_grid['HILL_M'].iloc[0]
    mountain_prop = morpho_grid['MOUNTAIN_M'].iloc[0]

    print(f"  Morphological proportions:")
    print(f"    Plain: {plain_prop:.4f}")
    print(f"    Hill: {hill_prop:.4f}")
    print(f"    Mountain: {mountain_prop:.4f}")
    print(f"    Sum: {plain_prop + hill_prop + mountain_prop:.4f}")

    # Get soil proportions
    soil_grid = soil_data[soil_data['GRID_OID'] == grid_id]
    if soil_grid.empty:
        print(f"  ⚠️  No soil data for grid {grid_id}")
        continue

    non_rock_prop = soil_grid['NON_ROCK_S'].iloc[0]
    rock_prop = soil_grid['ROCK_S'].iloc[0]

    print(f"  Soil proportions:")
    print(f"    Non-rock: {non_rock_prop:.4f}")
    print(f"    Rock: {rock_prop:.4f}")
    print(f"    Sum: {non_rock_prop + rock_prop:.4f}")

    # Get anthropization proportions
    anthro_grid = anthro_data[anthro_data['GRID_OID'] == grid_id]
    if anthro_grid.empty:
        print(f"  ⚠️  No anthropization data for grid {grid_id}")
        continue

    non_anthro_prop = anthro_grid['NON_ANTHROPISED'].iloc[0]
    anthro_prop = anthro_grid['ANTHROPISED'].iloc[0]

    print(f"  Anthropization proportions:")
    print(f"    Non-anthropised: {non_anthro_prop:.4f}")
    print(f"    Anthropised: {anthro_prop:.4f}")
    print(f"    Sum: {non_anthro_prop + anthro_prop:.4f}")

    # Calculate grid factor using current method
    grid_factor_current = (
        # Morphological factors
            plain_prop * cost_factors['k_morpho_plain'] +
            hill_prop * cost_factors['k_morpho_hill'] +
            mountain_prop * cost_factors['k_morpho_mountain'] +

            # Soil factors
            non_rock_prop * cost_factors['k_soil_non_rock'] +
            rock_prop * cost_factors['k_soil_rock'] +

            # Anthropization factors
            non_anthro_prop * cost_factors['k_anthro_non_anthropised'] +
            anthro_prop * cost_factors['k_anthro_anthropised']
    )

    print(f"  Grid factor calculation (CURRENT METHOD - ADDITIVE):")
    print(
        f"    Morphological component: {plain_prop * cost_factors['k_morpho_plain'] + hill_prop * cost_factors['k_morpho_hill'] + mountain_prop * cost_factors['k_morpho_mountain']:.4f}")
    print(
        f"    Soil component: {non_rock_prop * cost_factors['k_soil_non_rock'] + rock_prop * cost_factors['k_soil_rock']:.4f}")
    print(
        f"    Anthropization component: {non_anthro_prop * cost_factors['k_anthro_non_anthropised'] + anthro_prop * cost_factors['k_anthro_anthropised']:.4f}")
    print(f"    Total grid factor: {grid_factor_current:.4f}")

    # Alternative calculation methods
    print(f"  Alternative calculations:")

    # Method 1: Multiplicative approach
    morpho_factor = 1.0 + (plain_prop * cost_factors['k_morpho_plain'] + hill_prop * cost_factors[
        'k_morpho_hill'] + mountain_prop * cost_factors['k_morpho_mountain'])
    soil_factor = 1.0 + (non_rock_prop * cost_factors['k_soil_non_rock'] + rock_prop * cost_factors['k_soil_rock'])
    anthro_factor = 1.0 + (non_anthro_prop * cost_factors['k_anthro_non_anthropised'] + anthro_prop * cost_factors[
        'k_anthro_anthropised'])
    grid_factor_multiplicative = morpho_factor * soil_factor * anthro_factor
    print(f"    Multiplicative method: {grid_factor_multiplicative:.4f}")

    # Method 2: Weighted average approach
    morpho_base = 1.0
    soil_base = 1.0
    anthro_base = 1.0

    morpho_adjustment = plain_prop * cost_factors['k_morpho_plain'] + hill_prop * cost_factors[
        'k_morpho_hill'] + mountain_prop * cost_factors['k_morpho_mountain']
    soil_adjustment = non_rock_prop * cost_factors['k_soil_non_rock'] + rock_prop * cost_factors['k_soil_rock']
    anthro_adjustment = non_anthro_prop * cost_factors['k_anthro_non_anthropised'] + anthro_prop * cost_factors[
        'k_anthro_anthropised']

    grid_factor_base_plus_adjustments = morpho_base + morpho_adjustment + soil_adjustment + anthro_adjustment
    print(f"    Base + adjustments method: {grid_factor_base_plus_adjustments:.4f}")

    # Method 3: Scaled approach (divide by some factor)
    grid_factor_scaled = grid_factor_current / 100  # Example scaling
    print(f"    Scaled method (/100): {grid_factor_scaled:.4f}")

    # Weight by intersection proportion
    total_weighted_factor += grid_factor_current * intersection_prop
    total_weight += intersection_prop

# Calculate final geographical factor
if total_weight > 0:
    geo_factor = total_weighted_factor / total_weight
else:
    geo_factor = 1.0

geo_factor = max(geo_factor, 0.1)  # Ensure minimum factor of 0.1

print(f"\n3. FINAL RESULTS:")
print("-" * 50)
print(f"Total weighted factor: {total_weighted_factor:.4f}")
print(f"Total weight: {total_weight:.4f}")
print(f"Final geographical factor: {geo_factor:.4f}")

if geo_factor > 10:
    print(f"🚨 EXTREMELY HIGH FACTOR - CALCULATION ERROR CONFIRMED!")
    print(f"   Factor of {geo_factor:.1f} means {((geo_factor - 1) * 100):.0f}% cost increase")
    print(f"   This is unrealistic for terrain impact")
elif geo_factor > 3:
    print(f"⚠️  High factor - may indicate calculation issue")
elif geo_factor > 0.5 and geo_factor < 2.0:
    print(f"✓ Reasonable factor - terrain impact seems realistic")

print(f"\n4. RECOMMENDED FIX:")
print("-" * 50)
print(f"The current additive method produces unrealistic results.")
print(f"Consider one of these approaches:")
print(f"1. Use much smaller cost factor values (divide by 100-1000)")
print(f"2. Use multiplicative approach with smaller adjustments")
print(f"3. Redefine cost factors as percentage adjustments from baseline")
print(f"4. Use weighted combination instead of simple addition")

# Test with model's actual method
try:
    actual_geo_factor = model._calculate_geo_factor(
        massflow_kg_per_s,
        intersected_grids,
        intersected_proportions,
        morpho_data,
        soil_data,
        anthro_data
    )
    print(f"\n5. MODEL'S ACTUAL CALCULATION:")
    print("-" * 50)
    print(f"Model calculated factor: {actual_geo_factor:.4f}")
    print(f"Matches manual calculation: {abs(actual_geo_factor - geo_factor) < 0.001}")
except Exception as e:
    print(f"\n5. ERROR IN MODEL'S CALCULATION:")
    print("-" * 50)
    print(f"Error: {e}")