#!/usr/bin/env python3

"""
Plot hourly emission data for SILLA 2 node across all scenarios
Creates stacked area charts showing emission_pos and captured_emission for each scenario
Shows CO2 Abated Rates (operational emission abatement performance, not design capacity)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Try to import cmcrameri for better colors
try:
    import cmcrameri.cm as cmc

    navia_available = True
    print("✅ CMC colormaps loaded successfully!")
except ImportError:
    print("⚠️  cmcrameri not available. Using matplotlib colors as fallback.")
    navia_available = False


def load_silla2_data(excel_file):
    """
    Load SILLA 2 hourly emission data from Excel file
    """
    if not Path(excel_file).exists():
        print(f"❌ Excel file not found: {excel_file}")
        return None, None

    print(f"📂 Loading data from: {excel_file}")

    # Read all sheets
    try:
        excel_data = pd.read_excel(excel_file, sheet_name=None)
        print(f"✅ Found {len(excel_data)} sheets in Excel file")

        # Get summary data
        summary_df = excel_data.get('Summary', None)
        if summary_df is not None:
            print(f"📊 Summary sheet contains {len(summary_df)} scenarios")

        # Get scenario data (exclude Summary sheet)
        scenario_data = {sheet_name: df for sheet_name, df in excel_data.items()
                         if sheet_name != 'Summary'}

        print(f"📈 Loaded hourly data for {len(scenario_data)} scenarios")
        for scenario, df in scenario_data.items():
            print(f"   {scenario}: {len(df)} data points")

        return scenario_data, summary_df

    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return None, None


def get_subplot_position(scenario_name):
    """
    Determine subplot position based on scenario name
    First row: AF-UC-300, AF-UC-150, AF-UC-100
    Second row: AF-4M-300, AF-4M-150, AF-4M-100
    Third row: CF-UC-300, CF-UC-150, CF-UC-100
    Fourth row: CF-4M-300, CF-4M-150, CF-4M-100
    """
    # Convert Excel sheet names back to scenario names
    scenario_name = scenario_name.replace('_', '-')

    # Define the grid layout mapping (4 rows x 3 columns)
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


def create_time_axis(num_hours=8760):
    """
    Create time axis with proper labels for a year
    """
    # Create hour array
    hours = np.arange(num_hours)

    # Define month boundaries (approximate)
    month_boundaries = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_centers = [(month_boundaries[i] + month_boundaries[i + 1]) / 2 for i in range(12)]

    return hours, month_boundaries, month_names, month_centers


def setup_colors():
    """
    Setup consistent colors for the plots
    """
    if navia_available:
        # Use cmcrameri colors - SWAPPED COLORS
        colors = {
            'emission_pos': cmc.navia(0.2),  # Greenish for emissions released
            'captured_emission': cmc.navia(0.8),  # Bluish for captured emissions
            'total_line': 'black'  # Black line for total
        }
    else:
        # Fallback colors - SWAPPED COLORS
        colors = {
            'emission_pos': '#4ecdc4',  # Teal-green for emissions released
            'captured_emission': '#4a90e2',  # Blue for captured emissions
            'total_line': 'black'  # Black line for total
        }

    return colors


def create_scenario_subplot(ax, scenario_name, df, colors, month_boundaries, month_centers, month_names):
    """
    Create stacked area chart for a single scenario
    """
    hours = df['hour'].values
    emission_pos = df['emission_pos'].values
    captured_emission = df['captured_emission'].values
    produced_emission = df['produced_emission'].values

    # Calculate some statistics
    total_produced = np.sum(produced_emission)
    total_captured = np.sum(captured_emission)
    total_emission_pos = np.sum(emission_pos)
    capture_rate = (total_captured / total_produced * 100) if total_produced > 0 else 0  # Actually CO2 abatement rate

    # Create stacked area plot with swapped colors
    ax.fill_between(hours, 0, emission_pos,
                    color=colors['emission_pos'], alpha=0.8, label='Emissions Released')
    ax.fill_between(hours, emission_pos, emission_pos + captured_emission,
                    color=colors['captured_emission'], alpha=0.8, label='Emissions Captured')

    # Add total emission line for reference
    ax.plot(hours, produced_emission, color=colors['total_line'],
            linewidth=1, alpha=0.7, linestyle='--', label='Total Produced')

    # Set title with CO2 abated rate (rounded to nearest integer)
    scenario_display = scenario_name.replace('_', '-')
    ax.set_title(f'{scenario_display}\nCO2 Abated Rate: {capture_rate:.0f}%',
                 fontsize=10, weight='bold', pad=10)

    # Set x-axis with month labels
    ax.set_xlim(0, max(hours) if len(hours) > 0 else 8760)
    ax.set_xticks(month_centers)
    ax.set_xticklabels(month_names, fontsize=8)

    # Add vertical lines for month boundaries
    for boundary in month_boundaries[1:-1]:  # Skip first and last
        ax.axvline(x=boundary, color='gray', alpha=0.3, linewidth=0.5)

    # Set y-axis
    max_emission = max(np.max(produced_emission), np.max(emission_pos + captured_emission)) if len(
        produced_emission) > 0 else 100
    ax.set_ylim(0, max_emission * 1.05)
    ax.set_ylabel('Emissions (t/h)', fontsize=8)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add statistics text box
    stats_text = f'Annual Total:\n{total_produced:.0f} t'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return capture_rate  # Returns CO2 abatement rate percentage


def plot_silla2_emissions(scenario_data, summary_df=None, output_file='SILLA2_hourly_emissions_plots.png'):
    """
    Create subplot visualization of SILLA 2 hourly emissions for all scenarios
    """
    print(f"\n{'=' * 60}")
    print("📊 CREATING SILLA 2 HOURLY EMISSIONS PLOTS")
    print(f"{'=' * 60}")

    # Setup colors and time axis
    colors = setup_colors()
    hours, month_boundaries, month_names, month_centers = create_time_axis()

    # Create figure with subplots (4 rows, 3 columns)
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))

    # Adjust spacing between subplots (less bottom margin since no explanation text)
    plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.96, bottom=0.05, left=0.05, right=0.98)

    # Track statistics for CO2 abatement rate (% of total emissions that were captured)
    scenario_stats = {}
    plotted_scenarios = 0

    # Expected scenarios in order
    expected_scenarios = ['AF_UC_300', 'AF_UC_150', 'AF_UC_100',
                          'AF_4M_300', 'AF_4M_150', 'AF_4M_100',
                          'CF_UC_300', 'CF_UC_150', 'CF_UC_100',
                          'CF_4M_300', 'CF_4M_150', 'CF_4M_100']

    # Process each scenario
    for sheet_name in expected_scenarios:
        print(f"\n📊 Processing: {sheet_name}")

        # Get subplot position
        scenario_display = sheet_name.replace('_', '-')
        position = get_subplot_position(scenario_display)

        if position is None:
            print(f"⚠️  Unknown position for scenario {sheet_name}")
            continue

        row, col = position
        ax = axes[row, col]

        # Check if we have data for this scenario
        if sheet_name in scenario_data:
            df = scenario_data[sheet_name]

            # Validate data
            required_columns = ['hour', 'produced_emission', 'emission_pos', 'captured_emission']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"❌ Missing columns in {sheet_name}: {missing_columns}")
                ax.text(0.5, 0.5, f'{scenario_display}\n(Missing data)',
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_xlim(0, 8760)
                ax.set_ylim(0, 100)
            else:
                # Create the plot
                capture_rate = create_scenario_subplot(ax, sheet_name, df, colors,
                                                       month_boundaries, month_centers, month_names)
                scenario_stats[sheet_name] = capture_rate
                plotted_scenarios += 1
                print(f"✅ Plotted {sheet_name} (CO2 abated rate: {capture_rate:.0f}%)")
        else:
            print(f"⚠️  No data found for {sheet_name}")
            ax.text(0.5, 0.5, f'{scenario_display}\n(No data)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xlim(0, 8760)
            ax.set_ylim(0, 100)

        # Clean up subplot appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Main title removed as requested
    # fig.suptitle('SILLA 2 - Hourly Emission Profiles Across All Scenarios',
    #             fontsize=16, weight='bold', y=0.96)

    # Create legend (using first plotted subplot)
    if plotted_scenarios > 0:
        # Find first subplot with data
        legend_ax = None
        for i in range(4):
            for j in range(3):
                if axes[i, j].collections:  # Has plotted data
                    legend_ax = axes[i, j]
                    break
            if legend_ax:
                break

        if legend_ax:
            handles, labels = legend_ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
                       ncol=3, frameon=True, fancybox=True, shadow=True)

    # Explanatory text removed as requested
    # explanation = ("Stacked areas show hourly emissions throughout the year.\n"
    #               "Green: Emissions released to atmosphere | Blue: Emissions captured by CCS\n"
    #               "Dashed line: Total produced emissions")
    # fig.text(0.5, 0.06, explanation, ha='center', fontsize=10, style='italic')

    # Save the plot
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', format='png')
        print(f"💾 Saved plot as: {output_file}")
    except Exception as e:
        print(f"❌ Error saving plot: {e}")

    plt.show()

    # Print summary statistics
    if scenario_stats:
        print(f"\n{'=' * 60}")
        print("📊 CO2 ABATEMENT SUMMARY")
        print(f"{'=' * 60}")

        # Sort by CO2 abatement rate
        sorted_stats = sorted(scenario_stats.items(), key=lambda x: x[1], reverse=True)

        for scenario, capture_rate in sorted_stats:
            scenario_display = scenario.replace('_', '-')
            print(f"  {scenario_display:12}: {capture_rate:6.0f}%")

        avg_capture_rate = np.mean(list(scenario_stats.values()))
        print(f"\n  Average CO2 abated rate: {avg_capture_rate:.0f}%")
        print(f"  Scenarios plotted: {plotted_scenarios}/12")

    return plotted_scenarios


def main():
    """
    Main function to load data and create plots
    """
    print("SILLA 2 Hourly Emissions Visualization")
    print("=" * 60)

    # Define input file
    excel_file = 'SILLA2_hourly_emissions_all_scenarios.xlsx'

    # Load data
    scenario_data, summary_df = load_silla2_data(excel_file)

    if scenario_data is None:
        print("❌ Failed to load data. Make sure the Excel file exists.")
        print(f"   Expected file: {excel_file}")
        print("   Run the extraction script first to generate the data.")
        return

    if len(scenario_data) == 0:
        print("❌ No scenario data found in Excel file.")
        return

    # Create plots
    plotted_count = plot_silla2_emissions(scenario_data, summary_df)

    print(f"\n{'=' * 60}")
    print("🎯 VISUALIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"📊 Successfully plotted: {plotted_count}/12 scenarios")
    print(f"💾 Output file: SILLA2_hourly_emissions_plots.png")
    print("=" * 60)


if __name__ == "__main__":
    main()