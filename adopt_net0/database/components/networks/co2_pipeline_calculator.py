# CO2 Pipeline CAPEX Calculator
# Run this after setting up the CO2_Pipeline_CostModel environment

def calculate_co2_pipeline_capex(length_km, massflow_t_per_h):
    """
    Calculate CAPEX for CO2 pipeline given distance and mass flow

    Args:
        length_km: Pipeline length in kilometers
        massflow_t_per_h: Mass flow rate in tonnes per hour

    Returns:
        dict: CAPEX components and total cost
    """

    # Convert mass flow to kg/s
    massflow_kg_per_s = massflow_t_per_h / 3.6

    # Set up model options
    options = {
        "length_km": length_km,
        "massflow_min_kg_per_s": massflow_kg_per_s,
        "massflow_max_kg_per_s": massflow_kg_per_s,
        "massflow_evaluation_points": 1,
        "source": "Oeuvray",
        "timeframe": "mid-term",
        "terrain": "Onshore",  # or "Offshore"
        "electricity_price_eur_per_mw": 60.0,
        "operating_hours_per_a": 8000.0,
        "p_inlet_bar": 10.0,
        "p_outlet_bar": 70.0,
        "velocity_m_s": 5.0,
        # Geographical data (empty if not available)
        "morpho_data": pd.DataFrame(),
        "soil_data": pd.DataFrame(),
        "anthro_data": pd.DataFrame(),
        "intersected_grids": [],
        "intersected_proportions": []
    }

    # Initialize and run the model
    model = CO2_Pipeline_CostModel("CO2_Pipeline")
    results = model.calculate_indicators(options)

    # Extract CAPEX information
    capex_info = {
        "gamma1_fixed_cost": results["financial_indicators"]["gamma1"],
        "gamma2_variable_cost": results["financial_indicators"]["gamma2"],
        "total_capex": (results["financial_indicators"]["gamma1"] +
                        results["financial_indicators"]["gamma2"] * massflow_t_per_h),
        "opex_fixed": results["financial_indicators"]["opex_fixed"],
        "opex_variable": results["financial_indicators"]["opex_variable"],
        "levelized_cost": results["financial_indicators"]["levelized_cost"],
        "detailed_costs": results["costs_detailed"]
    }

    return capex_info


# Example usage for your specific case
if __name__ == "__main__":
    length_km = 10.38
    massflow_t_per_h = 400

    print(f"Calculating CAPEX for:")
    print(f"  Distance: {length_km} km")
    print(f"  Mass flow: {massflow_t_per_h} t/h")
    print()

    try:
        capex_results = calculate_co2_pipeline_capex(length_km, massflow_t_per_h)

        print("CAPEX Results:")
        print(f"  Fixed cost (γ₁): {capex_results['gamma1_fixed_cost']:,.0f} EUR")
        print(f"  Variable cost (γ₂): {capex_results['gamma2_variable_cost']:,.2f} EUR/(t/h)")
        print(f"  Total CAPEX: {capex_results['total_capex']:,.0f} EUR")
        print(f"  OPEX Fixed: {capex_results['opex_fixed']:.1%}")
        print(f"  OPEX Variable: {capex_results['opex_variable']:,.2f} EUR")
        print(f"  Levelized Cost: {capex_results['levelized_cost']:,.2f} EUR")

    except Exception as e:
        print(f"Error calculating CAPEX: {e}")
        print("Make sure all dependencies are installed and data files are available.")