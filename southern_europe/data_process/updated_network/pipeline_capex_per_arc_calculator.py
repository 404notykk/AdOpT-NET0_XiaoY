import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utilities/')))

try:
    from adopt_net0.database.components.networks.enhanced_co2_pipelines_cost_model import \
        CO2_Pipeline_CostModel as EnhancedModel
except ImportError as e:
    print(f"⚠️  Warning: Could not import EnhancedModel: {e}")
    print("   This is required for gamma calculations. Please check your adopt_net0 installation.")

# Import shared functions from arc_specific_functions
from arc_specific_functions import (
    load_network_data,
    load_intersection_data,
    get_node_emission,
    calculate_total_annual_emission,
    calculate_global_max_massflow,
    calculate_global_min_massflow,
    calculate_arc_gammas,
    create_gamma_matrices,
    create_zero_gamma_matrices,
    get_pipeline_length,
    validate_pipeline_transport,
    get_all_possible_arcs,
    create_base_options,
    add_geographical_options,
    print_section_header,
    save_to_excel,
    determine_arc_terrain
)


def main():
    """Main execution function"""

    print_section_header("FULL NETWORK ARC GAMMA CALCULATOR")
    print("This script calculates gamma1 and gamma2 values for every possible arc")
    print("in the network and saves them as matrices in an Excel file.")
    print("Gamma3 and gamma4 matrices will be created with zeros.")
    print("\n🌊 SPECIAL CONFIGURATION: Arc 13-14 will be processed as OFFSHORE terrain")
    print("🏞️  All other arcs will be processed as ONSHORE terrain")
    print("\n🔇 VERBOSE OUTPUT: Cost model calculations are suppressed for cleaner output")
    print("📊 UPDATED BEHAVIOR: Failed calculations will be set to 0 instead of NaN")

    try:
        # Load all data using shared function
        print("\n🔄 Loading network data...")
        data_dict = load_network_data("../../greece_data")

        # Verify data loaded correctly
        print(f"\n📊 Data verification:")
        print(f"   Nodes: {len(data_dict['network_nodes'])}")
        print(f"   Distance matrix: {data_dict['network_distance'].shape}")
        print(f"   Pipeline matrix: {data_dict['network_pipeline'].shape}")
        print(f"   Emission data: {len(data_dict['network_emission_flux'])}")

        # Check if annual_emission column exists
        if 'annual_emission' in data_dict['network_emission_flux'].columns:
            total_emission = data_dict['network_emission_flux']['annual_emission'].sum()
            print(f"   Total annual emissions: {total_emission:,.0f} kg/year")
        else:
            print(f"   ⚠️  Warning: 'annual_emission' column not found in emission data")
            print(f"   Available columns: {list(data_dict['network_emission_flux'].columns)}")

        # Load intersection data using shared function
        print("\n🔄 Loading intersection data...")
        intersection_file_path = Path("../../greece_data/geographical_feature/route_grid_intersections.xlsx")

        # Get all possible arcs to determine which intersection data to load
        possible_arcs = get_all_possible_arcs(data_dict['network_pipeline'])
        pipeline_names = [f"{from_node}_{to_node}" for from_node, to_node in possible_arcs]

        print(f"   Found {len(possible_arcs)} possible arcs")
        if len(possible_arcs) > 0:
            print(f"   Sample arcs: {possible_arcs[:5]}")  # Show first 5 arcs

        # Check for offshore arc
        offshore_arcs = [(f, t) for f, t in possible_arcs if determine_arc_terrain(f, t) == "Offshore"]
        if offshore_arcs:
            print(f"   🌊 Offshore arcs detected: {offshore_arcs}")
        else:
            print(f"   ℹ️  No offshore arcs found in this network")

        print(f"   Looking for intersection data for {len(pipeline_names)} pipeline combinations")

        intersection_data = load_intersection_data(intersection_file_path, pipeline_names)
        data_dict['intersection_data'] = intersection_data

        # Verify we have all required data
        if 'network_emission_flux' not in data_dict or 'annual_emission' not in data_dict[
            'network_emission_flux'].columns:
            print("⚠️  Warning: Annual emission data not found. Please check data loading.")
            print(f"   Available columns in emission data: {list(data_dict['network_emission_flux'].columns)}")
            return

        # Calculate and display global flow parameters
        print("\n🔄 Calculating global flow parameters...")

        # Calculate global max mass flow
        global_max_massflow_kg_s = calculate_global_max_massflow(data_dict['network_emission_flux'])

        # Calculate global min mass flow based on smallest emitting node
        global_min_massflow_kg_s = calculate_global_min_massflow(data_dict['network_emission_flux'])

        print(f"📊 Global flow parameters:")
        print(f"   Global max mass flow: {global_max_massflow_kg_s:.2f} kg/s")
        print(f"   Global min mass flow: {global_min_massflow_kg_s:.3f} kg/s (based on smallest emitting node)")

        print("\n🔄 Starting gamma calculations...")
        print("   Note: Verbose output from cost model is suppressed - only key progress will be shown")
        print("   Note: Failed calculations will be set to 0 instead of NaN")

        # Pass the terrain determination function to create_gamma_matrices
        gamma1_matrix, gamma2_matrix = create_gamma_matrices(data_dict, terrain_function=determine_arc_terrain)

        # Verify gamma calculations
        gamma1_nonzero = (gamma1_matrix != 0).sum().sum()
        gamma2_nonzero = (gamma2_matrix != 0).sum().sum()
        gamma1_total = gamma1_matrix.size
        gamma2_total = gamma2_matrix.size

        print(f"\n📊 Gamma calculation results:")
        print(f"   Gamma1 non-zero values: {gamma1_nonzero} out of {gamma1_total} total cells")
        print(f"   Gamma2 non-zero values: {gamma2_nonzero} out of {gamma2_total} total cells")
        print(f"   Success rate: {(gamma1_nonzero / len(possible_arcs) * 100):.1f}%" if len(
            possible_arcs) > 0 else "   Success rate: N/A")

        if gamma1_nonzero == 0 or gamma2_nonzero == 0:
            print("⚠️  Warning: No valid gamma values calculated. Please check:")
            print("   - Network distance data")
            print("   - Pipeline transport matrix")
            print("   - Emission data")
            print("   - Enhanced cost model import")
        else:
            print("✅ Gamma calculations completed successfully")

        # Create gamma3 and gamma4 matrices filled with zeros
        print("\n🔄 Creating gamma3 and gamma4 matrices...")
        gamma3_matrix, gamma4_matrix = create_zero_gamma_matrices(gamma1_matrix)

        # Set output path to network_capex_metrics folder
        output_dir = Path("../../greece_data/network_capex_metrics")
        output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        output_file = "capex_defined_per_arc.xlsx"
        full_output_path = output_dir / output_file

        print(f"\n📁 Excel file will be saved at: {full_output_path.absolute()}")

        # Save to Excel with all four matrices
        save_to_excel(gamma1_matrix, gamma2_matrix, gamma3_matrix, gamma4_matrix, filename=str(full_output_path))

        # Final summary
        print_section_header("ANALYSIS COMPLETE")
        print("✅ All gamma values have been calculated and saved to 'capex_defined_per_arc.xlsx'")
        print("📊 The Excel file contains four sheets: 'gamma1', 'gamma2', 'gamma3', and 'gamma4'")
        print("🗺️  Matrix format: rows = 'from' nodes, columns = 'to' nodes")
        print("🔢 Missing/failed calculations are set to 0 instead of NaN")
        print(f"📍 File location: {full_output_path.absolute()}")
        print(f"📁 Saved in: greece_data/network_capex_metrics/")

        # Display terrain-specific statistics
        print(f"\n📈 Final statistics:")
        print(f"   Matrix dimensions: {gamma1_matrix.shape[0]} × {gamma1_matrix.shape[1]}")
        print(f"   Total possible connections: {gamma1_matrix.size}")
        print(f"   Total possible arcs: {len(possible_arcs)}")
        print(f"   Gamma1 calculated values: {gamma1_nonzero}")
        print(f"   Gamma2 calculated values: {gamma2_nonzero}")
        print(f"   Zero values: {len(possible_arcs) - gamma1_nonzero} arcs")

        # Show terrain breakdown
        onshore_arcs = [(f, t) for f, t in possible_arcs if determine_arc_terrain(f, t) == "Onshore"]
        offshore_arcs = [(f, t) for f, t in possible_arcs if determine_arc_terrain(f, t) == "Offshore"]

        print(f"\n🌍 Terrain breakdown:")
        print(f"   🏞️  Onshore arcs: {len(onshore_arcs)}")
        print(f"   🌊 Offshore arcs: {len(offshore_arcs)}")

        if offshore_arcs:
            print(f"   🌊 Offshore arc details: {offshore_arcs}")

        # Display flow parameter usage summary
        print(f"\n🔧 Flow parameter usage:")
        print(f"   Transport-only nodes (like node 10) used global min flow: {global_min_massflow_kg_s:.3f} kg/s")
        print(f"   Emission nodes used their own emission-based minimum flows")
        print(f"   All arcs used global max flow as upper bound: {global_max_massflow_kg_s:.2f} kg/s")

        # Show gamma value ranges for successful calculations
        if gamma1_nonzero > 0:
            # Flatten the matrices and get non-zero values
            gamma1_values = gamma1_matrix.values.flatten()
            gamma2_values = gamma2_matrix.values.flatten()

            gamma1_nonzero_values = gamma1_values[gamma1_values != 0]
            gamma2_nonzero_values = gamma2_values[gamma2_values != 0]

            # Calculate statistics
            gamma1_min = float(gamma1_nonzero_values.min())
            gamma1_max = float(gamma1_nonzero_values.max())
            gamma1_mean = float(gamma1_nonzero_values.mean())

            gamma2_min = float(gamma2_nonzero_values.min())
            gamma2_max = float(gamma2_nonzero_values.max())
            gamma2_mean = float(gamma2_nonzero_values.mean())

            print(f"\n📊 Gamma value ranges (non-zero values only):")
            print(f"   Gamma1: min={gamma1_min:,.0f}, max={gamma1_max:,.0f}, mean={gamma1_mean:,.0f} EUR")
            print(f"   Gamma2: min={gamma2_min:.3f}, max={gamma2_max:.3f}, mean={gamma2_mean:.3f} EUR/t")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

        # Additional debugging information
        print(f"\n🔍 Debugging information:")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Data path exists: {Path('../../greece_data').exists()}")
        print(
            f"   Excel file exists: {Path('../../greece_data/geographical_feature/node_metrics.xlsx').exists()}")
        print(f"   Output directory: {Path('../../greece_data/network_capex_metrics').exists()}")

        # Check if required modules are available
        try:
            from adopt_net0.database.components.networks.enhanced_co2_pipelines_cost_model import CO2_Pipeline_CostModel
            print(f"   Enhanced cost model: Available")
        except ImportError:
            print(f"   Enhanced cost model: NOT available (required for gamma calculations)")

        try:
            from southern_europe.data_process.utilities.defined_functions import calculate_annual_emission_values
            print(f"   Defined functions module: Available")
        except ImportError:
            print(f"   Defined functions module: NOT available")


if __name__ == "__main__":
    main()