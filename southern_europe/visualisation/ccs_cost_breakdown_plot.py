#!/usr/bin/env python3

"""
CCS Network Analysis - Cost Breakdown Visualization
Creates a stacked bar chart showing cost breakdown per ton CO2 captured for each scenario
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings

# Import cmcrameri for consistent color scheme
try:
    import cmcrameri.cm as cmc

    navia_available = True
    print("CMC colormaps loaded successfully!")
except ImportError:
    print("Warning: cmcrameri not available. Install with: pip install cmcrameri")
    print("Falling back to matplotlib colormaps")
    navia_available = False

warnings.filterwarnings('ignore')


def load_cost_breakdown_data():
    """
    Load cost breakdown data from Excel file
    """
    results_data_path = Path('..') / 'userData'
    excel_file_path = results_data_path / 'Overview.xlsx'

    if not excel_file_path.exists():
        print(f"❌ Excel file not found: {excel_file_path}")
        return None

    try:
        # Read the overview_model_based sheet
        df = pd.read_excel(excel_file_path, sheet_name='overview_model_based')

        print("🔍 Debug: Loading cost breakdown data...")
        print(f"📊 Sheet shape: {df.shape}")
        print(f"📊 Columns: {df.columns.tolist()}")

        # Extract scenario names from column headers (skip first 2 columns)
        scenario_names = list(df.columns[2:])
        print(f"🔍 Found scenarios: {scenario_names}")

        # Define the cost component rows based on the Excel structure
        cost_components = {
            'capture_capex': None,
            'capture_opex': None,
            'capture_electricity': None,
            'capture_heat': None,
            'transport_capex': None,
            'storage_opex': None,
            'storage_electricity': None
        }

        # Find the cost component rows based on exact search results
        # Using the exact row numbers found by the diagnostic script
        print("🔍 Using exact row positions found by diagnostic...")

        # Exact mapping based on diagnostic search results
        cost_component_rows = {
            'capture_capex': 50,  # Row 50: capture_capex
            'capture_opex': 51,  # Row 51: capture_opex
            'capture_electricity': 52,  # Row 52: capture_electricity
            'capture_heat': 53,  # Row 53: capture_heat
            'transport_capex': 54,  # Row 54: transport_capex
            'storage_opex': 55,  # Row 55: storage_opex
            'storage_electricity': 56  # Row 56: storage_electricity
        }

        # Verify these rows exist and have the expected names
        for component, row_idx in cost_component_rows.items():
            if row_idx < len(df):
                row_name = str(df.iloc[row_idx, 1]).strip() if pd.notna(df.iloc[row_idx, 1]) else ""
                cost_components[component] = row_idx
                print(f"    ✅ Found {component} at row {row_idx}: '{row_name}'")
            else:
                print(f"    ❌ Row {row_idx} for {component} not found in dataframe")

        print(f"🔍 Found cost component rows: {cost_components}")

        # Extract cost data for each scenario
        cost_data = {}

        for i, scenario_name in enumerate(scenario_names):
            col_idx = i + 2  # Column index (starting from 2)
            scenario_costs = {}

            print(f"\n🔍 DEBUG: Reading data for {scenario_name} (column {col_idx})")

            for component, row_idx in cost_components.items():
                if row_idx is not None:
                    cost_value = df.iloc[row_idx, col_idx]
                    row_name_in_col_b = str(df.iloc[row_idx, 1]) if pd.notna(df.iloc[row_idx, 1]) else "N/A"

                    print(f"  Row {row_idx}: {component} (Col B: '{row_name_in_col_b}') = {cost_value}")

                    if pd.notna(cost_value):
                        scenario_costs[component] = float(cost_value)
                    else:
                        scenario_costs[component] = 0.0
                else:
                    scenario_costs[component] = 0.0
                    print(f"  {component}: row not found, setting to 0")

            cost_data[scenario_name] = scenario_costs

            # Debug output
            total_cost = sum(scenario_costs.values())
            print(f"✅ {scenario_name}: Total = {total_cost:.2f} EUR/t CO2")

        print(f"✅ Successfully loaded cost data for {len(cost_data)} scenarios")
        return cost_data

    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_cost_breakdown_chart(cost_data):
    """
    Create a stacked bar chart showing cost breakdown per ton CO2 captured for each scenario
    """
    if not cost_data:
        print("❌ No cost data available")
        return

    # Prepare data for plotting
    scenarios = list(cost_data.keys())

    # Define cost components in logical order (bottom to top in stack)
    cost_components = [
        'capture_capex',
        'capture_opex',
        'capture_electricity',
        'capture_heat',
        'transport_capex',
        'storage_opex',
        'storage_electricity'
    ]

    # Create labels for better readability
    component_labels = {
        'capture_capex': 'Capture CAPEX',
        'capture_opex': 'Capture OPEX',
        'capture_electricity': 'Capture Electricity',
        'capture_heat': 'Capture Heat',
        'transport_capex': 'Transport CAPEX',
        'storage_opex': 'Storage OPEX',
        'storage_electricity': 'Storage Electricity'
    }

    # Extract values for each component across all scenarios
    component_values = {}
    for component in cost_components:
        component_values[component] = [cost_data[scenario].get(component, 0)
                                       for scenario in scenarios]

    # Set up colors using navia colormap if available
    if navia_available:
        # Use navia colormap for consistent, pleasant colors
        colors = [cmc.navia(i / len(cost_components)) for i in range(len(cost_components))]
    else:
        # Fallback to a custom color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create stacked bar chart
    bottom = np.zeros(len(scenarios))

    bars = []
    for i, component in enumerate(cost_components):
        values = component_values[component]
        bar = ax.bar(scenarios, values, bottom=bottom,
                     label=component_labels[component],
                     color=colors[i % len(colors)],
                     alpha=0.8, edgecolor='white', linewidth=0.5)
        bars.append(bar)
        bottom += values

    # Customize the plot
    ax.set_title('CCS Network Cost Breakdown per Ton CO₂ Captured by Scenario',
                 fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Scenario', fontsize=12, weight='bold')
    ax.set_ylabel('Unit Cost of Captured CO2', fontsize=12, weight='bold')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add legend with components ordered to match stacking (storage -> transport -> capture from top to bottom)
    legend_elements = [
        # Storage domain components (top of stack)
        plt.Line2D([0], [0], color=colors[6], linewidth=8, label='Storage Electricity'),
        plt.Line2D([0], [0], color=colors[5], linewidth=8, label='Storage OPEX'),
        # Transport domain component (middle of stack)
        plt.Line2D([0], [0], color=colors[4], linewidth=8, label='Transport CAPEX'),
        # Capture domain components (bottom of stack)
        plt.Line2D([0], [0], color=colors[3], linewidth=8, label='Capture Heat'),
        plt.Line2D([0], [0], color=colors[2], linewidth=8, label='Capture Electricity'),
        plt.Line2D([0], [0], color=colors[1], linewidth=8, label='Capture OPEX'),
        plt.Line2D([0], [0], color=colors[0], linewidth=8, label='Capture CAPEX'),
    ]

    # Create single legend with all components
    legend = ax.legend(handles=legend_elements,
                       bbox_to_anchor=(1.05, 1), loc='upper left',
                       frameon=True, fancybox=True, shadow=True,
                       title='Cost Components by Domain')
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(12)

    # Add value labels on top of each bar (total cost per ton)
    for i, scenario in enumerate(scenarios):
        total_cost = sum(cost_data[scenario].values())
        if total_cost == 0:
            label_text = 'N/A'
            # Position N/A label at a small height above the x-axis
            label_y = max(bottom) * 0.05  # 5% of max height for visibility
        else:
            label_text = f'{total_cost:.1f}'
            label_y = total_cost + (total_cost * 0.02)

        ax.text(i, label_y, label_text,
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Improve layout
    plt.tight_layout()

    # Save the plot
    filename = 'ccs_cost_breakdown_per_ton_co2.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')
    print(f"💾 Saved cost breakdown chart as: {filename}")
    plt.show()

    # Print summary statistics
    print(f"\n" + "=" * 60)
    print(f"📊 COST BREAKDOWN SUMMARY (EUR/t CO₂ captured)")
    print("=" * 60)

    for scenario in scenarios:
        total_cost = sum(cost_data[scenario].values())
        if total_cost == 0:
            print(f"\n{scenario}: N/A (No CCS deployment)")
        else:
            print(f"\n{scenario}: {total_cost:.2f} EUR/t CO₂")

            # Show percentage breakdown
            for component in cost_components:
                value = cost_data[scenario].get(component, 0)
                if value > 0:
                    percentage = (value / total_cost) * 100
                    print(f"  {component_labels[component]}: {value:.2f} EUR/t CO₂ ({percentage:.1f}%)")

    # Calculate average component costs across scenarios with non-zero costs
    scenarios_with_costs = [s for s in scenarios if sum(cost_data[s].values()) > 0]
    if scenarios_with_costs:
        print(f"\n📈 AVERAGE COMPONENT COSTS ACROSS SCENARIOS WITH CCS DEPLOYMENT:")
        num_scenarios_with_costs = len(scenarios_with_costs)
        for component in cost_components:
            avg_component = sum(
                cost_data[scenario].get(component, 0) for scenario in scenarios_with_costs) / num_scenarios_with_costs
            if avg_component > 0:
                print(f"  {component_labels[component]}: {avg_component:.2f} EUR/t CO₂")

    print("=" * 60)


def main():
    """
    Main function to create cost breakdown visualization
    """
    print("CCS Network Cost Breakdown Analysis (EUR/t CO₂ captured)")
    print("=" * 60)

    # Load cost breakdown data from Excel
    print("📋 Loading cost breakdown data from Excel...")
    cost_data = load_cost_breakdown_data()

    if not cost_data:
        print("❌ Failed to load cost data. Exiting.")
        return

    # Create cost breakdown chart
    print("\n📊 Creating cost breakdown visualization...")
    create_cost_breakdown_chart(cost_data)

    print("\n✅ Cost breakdown analysis complete!")


if __name__ == "__main__":
    main()