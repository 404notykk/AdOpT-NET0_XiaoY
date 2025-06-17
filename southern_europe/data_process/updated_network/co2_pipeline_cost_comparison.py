import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from adopt_net0.database.components.networks.co2_pipelines_cost_model import CO2_Pipeline_CostModel
from adopt_net0.database.components.networks.utilities.co2_pipelines_oeuvray import CO2Chain_Oeuvray


def plot_co2_pipeline_cost_model():
    """
    Plots CO2 pipeline cost model comparing linear fit vs actual CAPEX calculations.

    This script ensures calculation consistency by:
    1. Using CO2_Pipeline_CostModel.calculate_indicators() for the linear fit parameters (gamma1, gamma2)
    2. Using CO2Chain_Oeuvray.calculate_cost() for individual point calculations (same method used
       internally by CO2_Pipeline_CostModel)
    3. Applying the exact same correction factors as CO2_Pipeline_CostModel uses internally

    Note: Large cost jumps between certain mass flow rates may indicate:
    1. Discrete pipeline diameter sizing (standard pipe sizes)
    2. Compressor capacity thresholds (requiring larger/additional compressors)
    3. Pressure requirement changes at higher flow rates
    4. Equipment sizing steps in the underlying CO2Chain_Oeuvray model

    The cost breakdown (Pipeline vs Compression CAPEX) helps identify which component
    is driving sudden cost increases.
    """
    # Create instance of the cost model
    model = CO2_Pipeline_CostModel("CO2_Pipeline")

    # Set options with all required parameters
    options = {
        "length_km": 39.46,
        "currency_out": "EUR",
        "financial_year_out": 2024,
        "discount_rate": 0.1
    }

    # Calculate indicators to get the fitted linear model parameters
    results = model.calculate_indicators(options)

    # Print the results
    print("Financial Indicators:")
    for key, value in results["financial_indicators"].items():
        print(f"  {key}: {value}")

    print("\nTechnical Indicators:")
    for key, value in results["technical_indicators"].items():
        print(f"  {key}: {value}")

    # Get mass flow range from the model's options
    massflow_min_kg_per_s = model.options["massflow_min_kg_per_s"]
    massflow_max_kg_per_s = model.options["massflow_max_kg_per_s"]

    # Convert to t/h for plotting
    massflow_min_t_per_h = massflow_min_kg_per_s / 1000 * 3600
    massflow_max_t_per_h = massflow_max_kg_per_s / 1000 * 3600

    # Define specific mass flow rates for actual CAPEX calculation (in t/h)
    # Adding more granular points around the problematic 24-27 t/h range
    specific_massflow_t_per_h = [12, 15, 18, 21, 22, 23, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 30, 33, 36]

    # Filter to keep only values within the valid range
    specific_massflow_t_per_h = [mf for mf in specific_massflow_t_per_h
                                 if massflow_min_t_per_h <= mf <= massflow_max_t_per_h]

    # Convert back to kg/s for calculation
    specific_massflow_kg_per_s = [mf / 3600 * 1000 for mf in specific_massflow_t_per_h]

    # Calculate ACTUAL costs using the same methods as CO2_Pipeline_CostModel
    actual_costs = []
    actual_massflow_t_per_h = []
    calculation_module = CO2Chain_Oeuvray()

    print(f"\nCalculating actual CAPEX values for {len(specific_massflow_kg_per_s)} specific points...")
    print("Note: Using CO2Chain_Oeuvray.calculate_cost() with same correction factors as CO2_Pipeline_CostModel")

    for i, massflow_kg_per_s in enumerate(specific_massflow_kg_per_s):
        try:
            # Create options exactly as CO2_Pipeline_CostModel does internally
            temp_options = {
                "length_km": options["length_km"],
                "massflow_kg_per_s": massflow_kg_per_s,
                "timeframe": model.options["timeframe"],
                "terrain": model.options["terrain"],
                "electricity_price_eur_per_mw": model.options["electricity_price_eur_per_mw"],
                "operating_hours_per_a": model.options["operating_hours_per_a"],
                "p_inlet_bar": model.options["p_inlet_bar"],
                "p_outlet_bar": model.options["p_outlet_bar"],
                "discount_rate": options["discount_rate"]
            }

            # Calculate cost using CO2Chain_Oeuvray (same as CO2_Pipeline_CostModel does internally)
            cost_result = calculation_module.calculate_cost(temp_options)

            # Apply the same correction factors as CO2_Pipeline_CostModel uses internally
            cr_pipeline = (
                    options["discount_rate"]
                    * (1 + options["discount_rate"]) ** calculation_module.universal_data["z_pipe"]
                    / ((1 + options["discount_rate"]) ** calculation_module.universal_data["z_pipe"] - 1)
            )
            cr_compressor = (
                    options["discount_rate"]
                    * (1 + options["discount_rate"]) ** calculation_module.universal_data["z_pumpcomp"]
                    / ((1 + options["discount_rate"]) ** calculation_module.universal_data["z_pumpcomp"] - 1)
            )
            correction_factor = cr_pipeline / cr_compressor

            # Calculate components exactly as CO2_Pipeline_CostModel does
            pipeline_capex = cost_result["cost_pipeline"]["unit_capex"] * correction_factor
            compression_capex = cost_result["cost_compression"]["unit_capex"]
            capex_total = pipeline_capex + compression_capex

            # Apply currency conversion to match the fitted model parameters
            # (CO2_Pipeline_CostModel converts gamma1/gamma2, so individual points should be converted too)
            from adopt_net0.database.utilities import convert_currency
            capex_total_converted = convert_currency(
                capex_total,
                2024,  # CO2Chain_Oeuvray uses EUR 2024
                options["financial_year_out"],
                "EUR",  # CO2Chain_Oeuvray uses EUR
                options["currency_out"]
            )
            pipeline_capex_converted = convert_currency(
                pipeline_capex,
                2024,
                options["financial_year_out"],
                "EUR",
                options["currency_out"]
            )
            compression_capex_converted = convert_currency(
                compression_capex,
                2024,
                options["financial_year_out"],
                "EUR",
                options["currency_out"]
            )

            actual_costs.append(capex_total_converted)
            actual_massflow_t_per_h.append(massflow_kg_per_s / 1000 * 3600)

            print(
                f"  Point {i + 1}/{len(specific_massflow_kg_per_s)}: {massflow_kg_per_s / 1000 * 3600:.1f} t/h → Total: {capex_total_converted:,.0f} {options['currency_out']} (Pipeline: {pipeline_capex_converted:,.0f}, Compression: {compression_capex_converted:,.0f})")

        except Exception as e:
            print(f"Error calculating cost for massflow {massflow_kg_per_s:.2f} kg/s: {e}")
            continue

    # Create the plot
    plt.figure(figsize=(12, 8))

    # 1. Plot the fitted linear model line
    x_range = np.linspace(massflow_min_t_per_h * 0.9, massflow_max_t_per_h * 1.1, 100)
    y_fitted = results["financial_indicators"]["gamma1"] + results["financial_indicators"]["gamma2"] * x_range
    plt.plot(x_range, y_fitted, 'b-', linewidth=2.5,
             label='Fitted linear model (γ₁ + γ₂×massflow)', alpha=0.9)

    # 2. Plot the ACTUAL calculated discrete points as scatter (not connected)
    if actual_costs:
        plt.scatter(actual_massflow_t_per_h, actual_costs, color='red', s=80,
                    label=f'Actual CAPEX calculations ({len(actual_costs)} points)',
                    zorder=5, edgecolors='darkred', alpha=0.8)

        # Calculate R² for the fit
        fitted_values = [results["financial_indicators"]["gamma1"] + results["financial_indicators"]["gamma2"] * mf
                         for mf in actual_massflow_t_per_h]

        ss_res = sum((actual - fitted) ** 2 for actual, fitted in zip(actual_costs, fitted_values))
        ss_tot = sum((actual - np.mean(actual_costs)) ** 2 for actual in actual_costs)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        r_squared = 0

    # Formatting
    plt.xlabel('Mass flow rate (t/h)', fontsize=14, fontweight='bold')
    plt.ylabel(f'Total CAPEX ({options["currency_out"]})', fontsize=14, fontweight='bold')
    plt.title(
        f'CO2 Pipeline Cost Model - Linear Fit vs Actual CAPEX\nLength: {options["length_km"]} km, Terrain: {model.options["terrain"]}',
        fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)

    # Add gamma values and other info to the plot
    info_text = f"""Linear Model Parameters:
γ₁ = {results['financial_indicators']['gamma1']:,.0f} {options['currency_out']}
γ₂ = {results['financial_indicators']['gamma2']:,.2f} {options['currency_out']}/(t/h)"""

    if actual_costs:
        info_text += f"""
R² = {r_squared:.4f}
Points calculated: {len(actual_costs)}"""

    plt.text(0.98, 0.02, info_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='gray'))

    # Format y-axis to show values in millions
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x / 1e6:.2f}M'))

    # Improve tick formatting
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig('co2_pipeline_cost_model.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print detailed analysis
    print(f"\nDetailed Model Analysis:")
    print(f"=" * 50)
    print(f"Pipeline length: {options['length_km']} km")
    print(f"Terrain type: {model.options['terrain']}")
    print(f"Mass flow range: {massflow_min_kg_per_s:.2f} - {massflow_max_kg_per_s:.2f} kg/s")
    print(f"Mass flow range: {massflow_min_t_per_h:.1f} - {massflow_max_t_per_h:.1f} t/h")
    print(f"\nLinear fit equation:")
    print(
        f"CAPEX = {results['financial_indicators']['gamma1']:,.0f} + {results['financial_indicators']['gamma2']:,.2f} × massflow(t/h) [{options['currency_out']}]")

    if actual_costs:
        print(f"R² = {r_squared:.6f}")
        print(f"Number of actual calculations: {len(actual_costs)}")

        # Calculate cost difference at min and max flow
        cost_at_min = min(actual_costs)
        cost_at_max = max(actual_costs)
        massflow_at_min = actual_massflow_t_per_h[actual_costs.index(cost_at_min)]
        massflow_at_max = actual_massflow_t_per_h[actual_costs.index(cost_at_max)]

        print(f"\nCost Analysis:")
        print(f"CAPEX at min flow ({massflow_at_min:.1f} t/h): {cost_at_min:,.0f} {options['currency_out']}")
        print(f"CAPEX at max flow ({massflow_at_max:.1f} t/h): {cost_at_max:,.0f} {options['currency_out']}")
        print(f"Total cost increase: {cost_at_max - cost_at_min:,.0f} {options['currency_out']}")
        print(f"Relative cost increase: {((cost_at_max / cost_at_min - 1) * 100):.1f}%")

        # Calculate average deviation between actual and fitted
        fitted_values = [results["financial_indicators"]["gamma1"] + results["financial_indicators"]["gamma2"] * mf
                         for mf in actual_massflow_t_per_h]
        avg_deviation = np.mean([abs(actual - fitted) / actual * 100
                                 for actual, fitted in zip(actual_costs, fitted_values)])
        print(f"Average deviation (actual vs fitted): {avg_deviation:.2f}%")

        # Analyze cost jumps between consecutive points
        print(f"\nCost Jump Analysis:")
        print(f"=" * 30)
        for i in range(1, len(actual_costs)):
            cost_diff = actual_costs[i] - actual_costs[i - 1]
            flow_diff = actual_massflow_t_per_h[i] - actual_massflow_t_per_h[i - 1]
            relative_change = (cost_diff / actual_costs[i - 1]) * 100
            cost_per_unit_flow = cost_diff / flow_diff if flow_diff > 0 else 0

            print(f"From {actual_massflow_t_per_h[i - 1]:.1f} to {actual_massflow_t_per_h[i]:.1f} t/h:")
            print(f"  Cost change: {cost_diff:+,.0f} {options['currency_out']} ({relative_change:+.1f}%)")
            print(f"  Cost per unit flow increase: {cost_per_unit_flow:,.0f} {options['currency_out']}/(t/h)")
            print()


if __name__ == "__main__":
    plot_co2_pipeline_cost_model()

'''
The behavior we're seeing is actually realistic pipeline engineering behavior 
rather than the smooth convex curve you'd expect from typical economic scaling relationships.
Looking at the detailed output, the massive jump occurs exactly between 26.0 and 26.5 t/h:

At 26.0 t/h: Pipeline: 4,808,475 EUR, Compression: 4,592,657 EUR, Total: 9,401,131 EUR
At 26.5 t/h: Pipeline: 7,744,022 EUR, Compression: 3,541,027 EUR, Total: 11,285,049 EUR

Pipeline cost jumps 61% while compression cost actually decreases 23%!

1. Pipe Diameter Threshold
At ~26.5 t/h, the algorithm hits a standard pipe size threshold. The CO2Chain_Oeuvray model uses discrete standard pipe diameters (NPS - Nominal Pipe Size), 
not continuous sizing. When flow exceeds the capacity of one standard size, it must jump to the next larger standard size.

2. Steel Grade Optimization
The output shows different steel grades being selected:
Lower flows: X65, X70, X80, X90, X100
After the jump: Consistently X52 (different optimization result)

3. Pressure-Diameter Coupling
The algorithm optimizes pressure and diameter together. 
A larger diameter allows lower pressures, which can actually reduce compression requirements 
(explaining why compression cost drops after the jump).
'''
