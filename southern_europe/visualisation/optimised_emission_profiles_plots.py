#!/usr/bin/env python3

"""
Enhanced CCS Network Analysis - Emission Profile Multi-Scenario Subplots
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
from collections import defaultdict

# Import cmcrameri for navia colormap (matching example script)
try:
    import cmcrameri.cm as cmc

    navia_available = True
    print("CMC colormaps loaded successfully!")
except ImportError:
    print("Warning: cmcrameri not available. Install with: pip install cmcrameri")
    print("Falling back to matplotlib's viridis colormap")
    navia_available = False

warnings.filterwarnings('ignore')


def load_scenario_data():
    """
    Load scenario information from Excel file including individual scenario sheets
    """
    results_data_path = Path('..') / 'userData'
    excel_file_path = results_data_path / 'Overview.xlsx'

    if not excel_file_path.exists():
        print(f"❌ Excel file not found: {excel_file_path}")
        return None, None

    try:
        # Read the overview_model_based sheet
        df = pd.read_excel(excel_file_path, sheet_name='overview_model_based')

        print("🔍 Debug: Extracting scenarios from known structure...")

        # The actual scenario names are in the column headers (AF-UC-300, AF-UC-150, etc.)
        scenario_names = list(df.columns[2:])  # Skip first 2 columns, get AF-UC-300, AF-UC-150, etc.

        print(f"🔍 Found scenario names in headers: {scenario_names}")

        scenarios = {}

        # Extract data starting from column 2 (where actual scenario data begins)
        for i, scenario_name in enumerate(scenario_names):
            col_idx = i + 2  # Column index (starting from 2)

            # Get h5 file from row 3
            h5_val = df.iloc[3, col_idx]

            print(f"🔍 Column {col_idx} ({scenario_name}): h5='{h5_val}'")

            if pd.notna(h5_val):
                h5_file = str(h5_val).strip()

                # Only add if h5 file is meaningful
                if h5_file and h5_file != 'nan':
                    scenarios[scenario_name] = h5_file
                    print(f"✅ Added scenario: {scenario_name} -> {h5_file}")
                else:
                    print(f"⚠️  Skipped {scenario_name}: invalid h5 file")
            else:
                print(f"⚠️  Skipped {scenario_name}: h5 file is NaN")

        if len(scenarios) == 0:
            print("❌ No valid scenarios found after processing")
            return None, None

        print(f"✅ Successfully loaded {len(scenarios)} scenarios from Excel file:")
        for scenario, h5_file in scenarios.items():
            print(f"  {scenario}: {h5_file}")

        # Now load individual scenario sheets for emission data
        print("\n🔍 Loading scenario-specific emission data...")
        scenario_data = {}

        for scenario_name in scenarios.keys():
            try:
                scenario_df = pd.read_excel(excel_file_path, sheet_name=scenario_name)

                # Process emission data by node (aggregate for nodes with multiple emitters)
                # but keep track of original order
                node_data = {}
                node_order = {}  # Track first appearance order

                # Debug: Track aggregation for node 12
                node_12_debug = []

                for idx, row in scenario_df.iterrows():
                    node_id = row['node_id']
                    node_name = row['node_name']
                    node_type = row['node_type']
                    emission_captured = row.get('emission_captured', 0)
                    emission_pos = row.get('emission_pos', 0)
                    emission_produced = row.get('emission_produced', 0)
                    capture_rate = row.get('capture_rate', 0)

                    # Skip rows with invalid node_id (NaN, None, or non-numeric)
                    if pd.isna(node_id) or node_id is None:
                        continue

                    # Convert node_id to integer to avoid decimal points
                    try:
                        node_id = int(float(node_id))  # Convert to int, handling potential float format
                    except (ValueError, TypeError):
                        continue  # Skip if can't convert to integer

                    # Debug: Track node 12 emitters for verification
                    if node_id == 12:
                        node_12_debug.append({
                            'row': idx,
                            'emission_captured': emission_captured,
                            'emission_pos': emission_pos,
                            'emission_produced': emission_produced,
                            'capture_rate': capture_rate
                        })

                    # If node already exists, aggregate the emissions
                    if node_id in node_data:
                        node_data[node_id]['emission_captured'] += emission_captured
                        node_data[node_id]['emission_pos'] += emission_pos
                        node_data[node_id]['emission_produced'] += emission_produced
                        # For capture rate, we'll recalculate it after aggregation
                    else:
                        # Track the first appearance order of this node
                        node_order[node_id] = idx
                        node_data[node_id] = {
                            'node_name': node_name,
                            'node_type': node_type,
                            'emission_captured': emission_captured,
                            'emission_pos': emission_pos,
                            'emission_produced': emission_produced,
                            'capture_rate': capture_rate,
                            'original_order': idx  # First appearance order
                        }

                # Recalculate capture rates for aggregated nodes
                for node_id, data in node_data.items():
                    total_produced = data['emission_produced']
                    if total_produced > 0:
                        data['capture_rate'] = data['emission_captured'] / total_produced
                    else:
                        data['capture_rate'] = 0

                # Debug: Print node 12 aggregation details if it exists
                if node_12_debug and 12 in node_data:
                    print(f"\n🔍 DEBUG - Node 12 aggregation in {scenario_name}:")
                    print(f"  Individual emitters found: {len(node_12_debug)}")
                    total_captured = sum([e['emission_captured'] for e in node_12_debug])
                    total_pos = sum([e['emission_pos'] for e in node_12_debug])
                    total_produced = sum([e['emission_produced'] for e in node_12_debug])

                    print(f"  Sum of emission_captured: {total_captured}")
                    print(f"  Sum of emission_pos: {total_pos}")
                    print(f"  Sum of emission_produced: {total_produced}")

                    final_data = node_data[12]
                    print(f"  Final aggregated values:")
                    print(f"    emission_captured: {final_data['emission_captured']}")
                    print(f"    emission_pos: {final_data['emission_pos']}")
                    print(f"    emission_produced: {final_data['emission_produced']}")
                    print(f"    capture_rate: {final_data['capture_rate']:.4f}")
                    print(
                        f"    calculated rate: {final_data['emission_captured'] / final_data['emission_produced']:.4f}")
                    print(
                        f"  ✅ Aggregation verified: {total_captured == final_data['emission_captured'] and total_pos == final_data['emission_pos']}")

                scenario_data[scenario_name] = node_data
                print(f"✅ Loaded emission data for {scenario_name}: {len(node_data)} nodes")

            except Exception as e:
                print(f"⚠️  Could not load scenario sheet {scenario_name}: {e}")
                scenario_data[scenario_name] = []

        return scenarios, scenario_data

    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_subplot_position(scenario_name):
    """
    Determine subplot position based on scenario name
    First row: AF-UC-300, AF-UC-150, AF-UC-100
    Second row: AF-4M-300, AF-4M-150, AF-4M-100
    Third row: CF-UC-300, CF-UC-150, CF-UC-100
    Fourth row: CF-4M-300, CF-4M-150, CF-4M-100
    """
    position_map = {
        # First row: AF-UC scenarios
        'AF-UC-300': (0, 0), 'AF-UC-150': (0, 1), 'AF-UC-100': (0, 2),
        # Second row: AF-4M scenarios
        'AF-4M-300': (1, 0), 'AF-4M-150': (1, 1), 'AF-4M-100': (1, 2),
        # Third row: CF-UC scenarios
        'CF-UC-300': (2, 0), 'CF-UC-150': (2, 1), 'CF-UC-100': (2, 2),
        # Fourth row: CF-4M scenarios
        'CF-4M-300': (3, 0), 'CF-4M-150': (3, 1), 'CF-4M-100': (3, 2),
    }

    return position_map.get(scenario_name, None)


def create_emission_subplot(ax, scenario_data, scenario_name):
    """
    Create emission profile stacked bar chart for a given scenario
    """
    # Setup colors - Green-ish for captured, Blue-ish for uncaptured
    if navia_available:
        navia_cmap = cmc.navia
        captured_color = navia_cmap(0.8)  # Green-ish for captured
        uncaptured_color = navia_cmap(0.2)  # Blue-ish for uncaptured
    else:
        captured_color = '#2E8B57'  # Sea Green for captured (fallback)
        uncaptured_color = '#4682B4'  # Steel Blue for uncaptured (fallback)

    if scenario_name not in scenario_data or not scenario_data[scenario_name]:
        # Handle missing data
        ax.text(0.5, 0.5, f'{scenario_name}\n(No emission data)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{scenario_name}', fontsize=12, weight='bold', pad=10)
        # Remove ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    # Get node data for this scenario
    nodes_data = scenario_data[scenario_name]

    # Filter out nodes with zero total emissions (only include nodes with actual emissions)
    emission_nodes = {}
    for node_id, data in nodes_data.items():
        total_emissions = data['emission_captured'] + data['emission_pos']
        if total_emissions > 0:  # Only include nodes with non-zero total emissions
            emission_nodes[node_id] = data

    if not emission_nodes:
        # Handle case with no emission nodes
        ax.text(0.5, 0.5, f'{scenario_name}\n(No emission sources)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{scenario_name}', fontsize=12, weight='bold', pad=10)
        # Remove ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    # Keep original order from Excel sheet (no sorting by emission amount)
    sorted_nodes = sorted(emission_nodes.items(), key=lambda x: x[1]['original_order'])

    # Prepare data for plotting
    node_ids = [item[0] for item in sorted_nodes]  # Use node_id
    captured_emissions = [item[1]['emission_captured'] for item in sorted_nodes]
    uncaptured_emissions = [item[1]['emission_pos'] for item in sorted_nodes]
    capture_rates = [item[1]['capture_rate'] for item in sorted_nodes]

    # Create x-axis positions
    x_pos = np.arange(len(node_ids))

    # Create stacked bar chart
    bars_captured = ax.bar(x_pos, captured_emissions, color=captured_color,
                           label='Captured Emissions', alpha=0.8)
    bars_uncaptured = ax.bar(x_pos, uncaptured_emissions, bottom=captured_emissions,
                             color=uncaptured_color, label='Uncaptured Emissions', alpha=0.8)

    # Add capture rate percentages on top of bars
    for i, (captured, uncaptured, rate) in enumerate(zip(captured_emissions, uncaptured_emissions, capture_rates)):
        total_height = captured + uncaptured
        if total_height > 0:  # Only add text if there are emissions
            percentage_text = f'{rate * 100:.0f}%'
            ax.text(i, total_height + max(captured_emissions + uncaptured_emissions) * 0.02,
                    percentage_text, ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Customize the plot
    ax.set_title(f'{scenario_name}', fontsize=12, weight='bold', pad=10)
    ax.set_xlabel('Node ID', fontsize=10)
    ax.set_ylabel('Emissions (t/year)', fontsize=10)  # Changed from tons/year to t/year

    # Set x-axis labels without rotation for straight text
    ax.set_xticks(x_pos)
    # Convert node_ids to integers for display (remove decimal points)
    node_id_labels = [str(int(node_id)) for node_id in node_ids]
    ax.set_xticklabels(node_id_labels, ha='center', fontsize=8, style='normal')

    # Format y-axis to use scientific notation for large numbers
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(8)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

    # Adjust layout to prevent label cutoff
    ax.margins(x=0.02)

    # Set y-axis limits with some padding
    max_emission = max([c + u for c, u in zip(captured_emissions, uncaptured_emissions)])
    if max_emission > 0:
        ax.set_ylim(0, max_emission * 1.15)

    print(f"✅ Created emission profile for {scenario_name}: {len(emission_nodes)} emission sources")


def create_emission_legends(fig):
    """
    Create legends for the emission profile plots
    """
    # Setup colors (same as in create_emission_subplot) - Green-ish for captured, Blue-ish for uncaptured
    if navia_available:
        navia_cmap = cmc.navia
        captured_color = navia_cmap(0.8)  # Green-ish for captured
        uncaptured_color = navia_cmap(0.2)  # Blue-ish for uncaptured
    else:
        captured_color = '#2E8B57'  # Sea Green for captured (fallback)
        uncaptured_color = '#4682B4'  # Steel Blue for uncaptured (fallback)

    # Create legend elements
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=captured_color, alpha=0.8,
                      edgecolor='black', label='Captured Emissions'),
        plt.Rectangle((0, 0), 1, 1, facecolor=uncaptured_color, alpha=0.8,
                      edgecolor='black', label='Uncaptured Emissions')
    ]

    # Add legend
    legend = fig.legend(handles=legend_elements, title='Emission Types',
                        loc='lower center', bbox_to_anchor=(0.5, 0.01),
                        frameon=True, fancybox=True, shadow=True, ncol=2)
    legend.get_title().set_fontweight('bold')

    return legend


def process_all_scenarios_emission_profiles(scenarios, scenario_data):
    """
    Process all scenarios and create emission profile subplot visualization
    """
    print(f"\n" + "=" * 60)
    print(f"📊 PROCESSING ALL SCENARIOS - EMISSION PROFILES")
    print("=" * 60)

    # Create figure with subplots (4 rows, 3 columns)
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Store scenario results for summary
    scenario_results = {}
    successful_scenarios = 0
    failed_scenarios = 0

    # Updated expected scenarios to match your Excel file
    expected_scenarios = ['AF-UC-300', 'AF-UC-150', 'AF-UC-100',
                          'AF-4M-300', 'AF-4M-150', 'AF-4M-100',
                          'CF-UC-300', 'CF-UC-150', 'CF-UC-100',
                          'CF-4M-300', 'CF-4M-150', 'CF-4M-100']

    # Process each expected scenario
    for scenario_name in expected_scenarios:
        print(f"\n📊 Processing: {scenario_name}")

        # Get subplot position
        position = get_subplot_position(scenario_name)
        if position is None:
            print(f"⚠️  Unknown scenario position for {scenario_name}, skipping...")
            failed_scenarios += 1
            continue

        row, col = position
        ax = axes[row, col]

        try:
            # Create emission profile visualization
            create_emission_subplot(ax, scenario_data, scenario_name)

            # Count emission sources for results
            emission_count = 0
            if scenario_name in scenario_data and scenario_data[scenario_name]:
                for node_data in scenario_data[scenario_name].values():
                    total_emissions = node_data['emission_captured'] + node_data['emission_pos']
                    if total_emissions > 0:  # Only count nodes with actual emissions
                        emission_count += 1

            scenario_results[scenario_name] = {'emission_sources': emission_count}
            successful_scenarios += 1
            print(f"✅ Successfully processed {scenario_name}")

        except Exception as e:
            print(f"❌ Error processing scenario {scenario_name}: {e}")
            ax.text(0.5, 0.5, f'{scenario_name}\n(Processing error)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'{scenario_name}', fontsize=12, weight='bold', pad=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
            failed_scenarios += 1

    # Create legend
    create_emission_legends(fig)

    # Adjust layout with room for legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, bottom=0.08, left=0.06, right=0.98, hspace=0.4, wspace=0.3)

    # Save with high quality
    filename = 'emission_profiles_multi_scenario.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')
    print(f"💾 Saved emission profiles plot as: {filename}")
    plt.show()

    # Print summary
    print(f"\n" + "=" * 70)
    print(f"🎯 EMISSION PROFILE ANALYSIS COMPLETE")
    print(f"📊 Expected scenarios: 12 (4×3 grid)")
    print(f"📊 Scenarios found in Excel: {len(scenarios)}")
    print(f"✅ Successfully processed: {successful_scenarios}")
    print(f"❌ Failed scenarios: {failed_scenarios}")

    if scenario_results:
        print(f"\n📋 SCENARIO EMISSION SUMMARY:")
        for scenario, results in scenario_results.items():
            print(f"  {scenario}: {results['emission_sources']} emission sources")

    print("=" * 70)

    return successful_scenarios, failed_scenarios


def main():
    """
    Main analysis function for emission profiles
    """
    print("Enhanced CO2 Transport Network Analysis - Emission Profile Subplots")
    print("=" * 70)

    # Load scenario data from Excel
    print("📋 Loading scenario information...")
    scenarios, scenario_data = load_scenario_data()
    if not scenarios or not scenario_data:
        print("❌ Failed to load scenarios. Exiting.")
        return

    # Process all scenarios for emission profiles
    successful_scenarios, failed_scenarios = process_all_scenarios_emission_profiles(
        scenarios, scenario_data)


if __name__ == "__main__":
    main()