import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from statsmodels import api as sm
import os
import math
from pathlib import Path

from adopt_net0.database.components.networks.utilities import *
from adopt_net0.database.utilities import convert_currency
from adopt_net0.database.data_component import DataComponent_CostModel


class CO2_Pipeline_CostModel(DataComponent_CostModel):
    """
    Enhanced CO2 Pipeline Cost Model with Geographical Factors

    Calculates CO2 transport costs and compression energy with terrain-based cost adjustments.

    Required geographical options:
    - morpho_data: DataFrame with morphological data (grid_id, proportion of plain/hill/mountain)
    - soil_data: DataFrame with soil data (grid_id, proportion of rock/non_rock)
    - anthro_data: DataFrame with anthropization data (grid_id, proportion of anthropised/non_anthropised)
    - intersected_grids: list of grid IDs that the pipeline intersects
    - intersected_proportions: list of proportions for each intersected grid
    """

    def __init__(self, tec_name):
        super().__init__(tec_name)

        # Default options
        self.default_options["source"] = "Oeuvray"
        self.default_options["timeframe"] = "mid-term"
        self.default_options["massflow_min_kg_per_s"] = 5.000
        self.default_options["massflow_max_kg_per_s"] = 10.000
        self.default_options["massflow_evaluation_points"] = 2
        self.default_options["terrain"] = "Offshore"
        self.default_options["electricity_price_eur_per_mw"] = 60.000
        self.default_options["operating_hours_per_a"] = 8000.000
        self.default_options["p_inlet_bar"] = 10.000
        self.default_options["p_outlet_bar"] = 70.000
        self.default_options["velocity_m_s"] = 5.000

        # Load cost factor table and fluid properties
        self.cost_factor_table = None
        self.co2_fluid_properties = None
        self._load_data()

    def _load_data(self):
        """Load cost factor table and CO2 fluid properties"""
        # Find the adopt_net0 directory by going up from current file location
        current_file = Path(__file__).resolve()

        # Go up directories until we find adopt_net0
        project_root = current_file
        while project_root.parent != project_root:  # Stop at filesystem root
            if (project_root / "adopt_net0").exists():
                break
            project_root = project_root.parent

        input_path = project_root / "adopt_net0" / "database" / "data" / "networks"

        # Load cost factor table
        cost_factor_path = input_path / "enhanced_co2_transport_cost_model" / "cost_factor_table.xlsx"
        if not cost_factor_path.exists():
            print(f"⚠️  Cost factor table not found at: {cost_factor_path}")
            print(f"Creating default cost factors...")
            # Create default cost factors if file doesn't exist
            self.cost_factor_table = self._create_default_cost_factors()
        else:
            try:
                self.cost_factor_table = pd.read_excel(cost_factor_path)
                print(f"✅ Loaded cost factor table from: {cost_factor_path}")
            except Exception as e:
                print(f"⚠️  Error loading cost factor table: {e}")
                self.cost_factor_table = self._create_default_cost_factors()

        # Load CO2 fluid properties
        co2_props_path = input_path / "co2_transport_oeuvray" / "CO2IsothermalProperties.xlsx"
        if co2_props_path.exists():
            try:
                self.co2_fluid_properties = pd.read_excel(co2_props_path, sheet_name=None)
                print(f"✅ Loaded CO2 fluid properties")
            except Exception as e:
                print(f"⚠️  Error loading CO2 properties: {e}")
                self.co2_fluid_properties = None

    def _create_default_cost_factors(self):
        """Create default cost factors if the Excel file is not available"""

        # Standard pipeline categories (DN in mm)
        categories = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200]

        cost_factors_data = []

        for dn in categories:
            # Create realistic cost factors that vary with pipeline size
            # These are incremental factors (0 = no change, positive = cost increase, negative = cost decrease)

            # Morphological factors
            k_morpho_plain = 0.000  # Plain terrain - no additional cost
            k_morpho_hill = 0.150 + (dn / 1000) * 0.050  # Hill terrain - moderate increase
            k_morpho_mountain = 0.300 + (dn / 1000) * 0.100  # Mountain terrain - significant increase

            # Soil factors
            k_soil_non_rock = 0.000  # Non-rock soil - no additional cost
            k_soil_rock = 0.200 + (dn / 1000) * 0.080  # Rock soil - increased excavation cost

            # Anthropization factors
            k_anthro_non_anthropised = 0.000  # Non-anthropized areas - no additional cost
            k_anthro_anthropised = 0.100 + (dn / 1000) * 0.050  # Anthropized areas - permits, restrictions

            cost_factors_data.append({
                'pipeline_category': dn,
                'k_morpho_plain': round(k_morpho_plain, 3),
                'k_morpho_hill': round(k_morpho_hill, 3),
                'k_morpho_mountain': round(k_morpho_mountain, 3),
                'k_soil_non_rock': round(k_soil_non_rock, 3),
                'k_soil_rock': round(k_soil_rock, 3),
                'k_anthro_non_anthropised': round(k_anthro_non_anthropised, 3),
                'k_anthro_anthropised': round(k_anthro_anthropised, 3)
            })

        df = pd.DataFrame(cost_factors_data)
        print(f"📊 Created default cost factors for {len(categories)} pipeline categories")
        return df

    def _estimate_operating_pressure(self, massflow_kg_per_s, terrain="Onshore"):
        """Estimate operating pressure based on mass flow rate"""
        if terrain == "Offshore":
            return 8.000
        else:
            if massflow_kg_per_s < 1:
                return 4.000
            elif massflow_kg_per_s < 10:
                return 6.000
            elif massflow_kg_per_s < 50:
                return 8.000
            else:
                return 10.000

    def _get_co2_density(self, pressure_mpa, temperature_k=288, terrain="Onshore"):
        """Get CO2 density at given pressure and temperature"""
        try:
            if self.co2_fluid_properties is not None:
                if terrain == "Offshore":
                    temp_sheet = "277K"
                else:
                    temp_sheet = "288K"

                if temp_sheet in self.co2_fluid_properties:
                    props = self.co2_fluid_properties[temp_sheet]
                    props = props.set_index("Pressure (MPa)")

                    if pressure_mpa in props.index:
                        return props.loc[pressure_mpa, "Density (kg/m3)"]
                    else:
                        return np.interp(pressure_mpa, props.index, props["Density (kg/m3)"])

            # Fallback to simple estimation
            if pressure_mpa >= 7.0:  # Liquid phase
                return 800.000
            else:  # Gas phase
                return 50.000 + pressure_mpa * 20.000

        except Exception:
            return 800.000 if pressure_mpa >= 7.0 else 100.000

    def _get_pipeline_category_from_massflow(self, massflow_kg_per_s, velocity_m_s=5,
                                             terrain="Onshore", operating_pressure_mpa=None):
        """
        Determine pipeline category (DN) based on mass flow rate, including CO2 density.

        Args:
            massflow_kg_per_s: Mass flow rate of CO2
            velocity_m_s: Design velocity
            terrain: "Onshore" or "Offshore"
            operating_pressure_mpa: Operating pressure (estimated if not provided)

        Returns:
            float: Exact pipeline DN (may be between standard categories)
        """
        if operating_pressure_mpa is None:
            operating_pressure_mpa = self._estimate_operating_pressure(massflow_kg_per_s, terrain)

        co2_density = self._get_co2_density(operating_pressure_mpa, terrain=terrain)

        # Calculate theoretical pipeline diameter in m, then convert to mm (DN)
        theoretical_diameter_m = 2 * math.sqrt(massflow_kg_per_s / (co2_density * velocity_m_s * math.pi))
        theoretical_DN = theoretical_diameter_m * 1000  # Convert to mm

        return theoretical_DN

    def _interpolate_cost_factors(self, target_DN):
        """
        Interpolate cost factors for a given DN using linear interpolation between adjacent categories

        Args:
            target_DN: Target pipeline DN (may be between standard categories)

        Returns:
            dict: Interpolated cost factors
        """
        if self.cost_factor_table is None or self.cost_factor_table.empty:
            print("⚠️  No cost factor table available, using zeros")
            return {
                'k_morpho_plain': 0.000,
                'k_morpho_hill': 0.000,
                'k_morpho_mountain': 0.000,
                'k_soil_non_rock': 0.000,
                'k_soil_rock': 0.000,
                'k_anthro_non_anthropised': 0.000,
                'k_anthro_anthropised': 0.000
            }

        available_categories = sorted(self.cost_factor_table['pipeline_category'].unique())

        # If exact match exists, return those factors
        if target_DN in available_categories:
            category_data = self.cost_factor_table[
                self.cost_factor_table['pipeline_category'] == target_DN
                ]
            return {
                'k_morpho_plain': round(category_data['k_morpho_plain'].iloc[0], 3),
                'k_morpho_hill': round(category_data['k_morpho_hill'].iloc[0], 3),
                'k_morpho_mountain': round(category_data['k_morpho_mountain'].iloc[0], 3),
                'k_soil_non_rock': round(category_data['k_soil_non_rock'].iloc[0], 3),
                'k_soil_rock': round(category_data['k_soil_rock'].iloc[0], 3),
                'k_anthro_non_anthropised': round(category_data['k_anthro_non_anthropised'].iloc[0], 3),
                'k_anthro_anthropised': round(category_data['k_anthro_anthropised'].iloc[0], 3)
            }

        # Find adjacent categories for interpolation
        lower_category = None
        upper_category = None

        for i, category in enumerate(available_categories):
            if category < target_DN:
                lower_category = category
            elif category > target_DN and upper_category is None:
                upper_category = category
                break

        # Handle edge cases
        if lower_category is None:
            # Target DN is smaller than smallest category, use smallest
            lower_category = available_categories[0]
            upper_category = available_categories[0]
        elif upper_category is None:
            # Target DN is larger than largest category, use largest
            lower_category = available_categories[-1]
            upper_category = available_categories[-1]

        # Get data for both categories
        lower_data = self.cost_factor_table[
            self.cost_factor_table['pipeline_category'] == lower_category
            ]
        upper_data = self.cost_factor_table[
            self.cost_factor_table['pipeline_category'] == upper_category
            ]

        # Calculate interpolation factor
        if upper_category == lower_category:
            t = 0  # No interpolation needed
        else:
            t = (target_DN - lower_category) / (upper_category - lower_category)

        # Interpolate each cost factor
        interpolated_factors = {}
        factor_columns = ['k_morpho_plain', 'k_morpho_hill', 'k_morpho_mountain',
                          'k_soil_non_rock', 'k_soil_rock', 'k_anthro_non_anthropised', 'k_anthro_anthropised']

        for col in factor_columns:
            lower_value = lower_data[col].iloc[0]
            upper_value = upper_data[col].iloc[0]
            interpolated_value = lower_value + t * (upper_value - lower_value)
            interpolated_factors[col] = round(interpolated_value, 3)

        return interpolated_factors

    def _calculate_incremental_geo_factor(self, massflow_kg_per_s, intersected_grids, intersected_proportions,
                                          morpho_data, soil_data, anthro_data):
        """
        Calculate geographical factor as INCREMENTAL cost adjustment.

        Returns: incremental factor (where updated_cost = original_cost * (1 + factor))
        """
        # Initialize grids processed counter
        grids_processed = 0

        # Debug information
        print(f"      🔍 Calculating geo factor for mass flow: {massflow_kg_per_s:.3f} kg/s")
        print(f"      🔍 Intersected grids: {len(intersected_grids)}")

        # Check if geographical data is available
        if (morpho_data.empty or soil_data.empty or anthro_data.empty or
                len(intersected_grids) == 0 or len(intersected_proportions) == 0):
            print(f"      ⚠️  No geographical data available")
            return 0.000

        velocity_m_s = self.options.get("velocity_m_s", 5.000)
        terrain = self.options.get("terrain", "Onshore")

        # Use inlet/outlet pressures to estimate operating pressure if available
        operating_pressure_mpa = None
        if "p_inlet_bar" in self.options and "p_outlet_bar" in self.options:
            operating_pressure_mpa = (self.options["p_inlet_bar"] + self.options[
                "p_outlet_bar"]) / 2 / 10  # Convert bar to MPa

        # Get exact pipeline DN and interpolated cost factors
        pipeline_DN = self._get_pipeline_category_from_massflow(
            massflow_kg_per_s, velocity_m_s, terrain, operating_pressure_mpa
        )
        cost_factors = self._interpolate_cost_factors(pipeline_DN)

        print(f"      🔍 Pipeline DN: {pipeline_DN:.1f} mm")
        print(f"      🔍 Cost factors: {cost_factors}")

        # Initialize weighted components (these will be the final weighted averages)
        total_weighted_morpho_factor = 0.000
        total_weighted_soil_factor = 0.000
        total_weighted_anthro_factor = 0.000

        # Process each intersected grid
        for grid_id, intersection_prop in zip(intersected_grids, intersected_proportions):
            # Validate intersection proportion
            if intersection_prop < 0 or intersection_prop > 1:
                print(f"      ⚠️  Invalid intersection proportion for grid {grid_id}: {intersection_prop}")
                continue

            # Try different grid ID formats for matching
            grid_id_variants = [grid_id, int(grid_id) if str(grid_id).isdigit() else grid_id, str(grid_id)]

            morpho_grid = pd.DataFrame()
            soil_grid = pd.DataFrame()
            anthro_grid = pd.DataFrame()

            # Find matches with different ID formats and column names
            for variant in grid_id_variants:
                if morpho_grid.empty:
                    # Try different possible column names for grid ID
                    for col_name in ['GRID_OID', 'grid_id', 'Grid_ID', 'ID', 'id']:
                        if col_name in morpho_data.columns:
                            morpho_grid = morpho_data[morpho_data[col_name] == variant]
                            if not morpho_grid.empty:
                                break

                if soil_grid.empty:
                    for col_name in ['GRID_OID', 'grid_id', 'Grid_ID', 'ID', 'id']:
                        if col_name in soil_data.columns:
                            soil_grid = soil_data[soil_data[col_name] == variant]
                            if not soil_grid.empty:
                                break

                if anthro_grid.empty:
                    for col_name in ['GRID_OID', 'grid_id', 'Grid_ID', 'ID', 'id']:
                        if col_name in anthro_data.columns:
                            anthro_grid = anthro_data[anthro_data[col_name] == variant]
                            if not anthro_grid.empty:
                                break

            # Skip if no data found for this grid
            if morpho_grid.empty or soil_grid.empty or anthro_grid.empty:
                print(f"      ⚠️  No data found for grid {grid_id}")
                continue

            grids_processed += 1

            # Get terrain proportions - try different possible column names
            try:
                # Morphological data
                plain_prop = 0.0
                hill_prop = 0.0
                mountain_prop = 0.0

                for col in morpho_grid.columns:
                    col_upper = str(col).upper()
                    if 'PLAIN' in col_upper:
                        plain_prop = float(morpho_grid[col].iloc[0])
                    elif 'HILL' in col_upper:
                        hill_prop = float(morpho_grid[col].iloc[0])
                    elif 'MOUNTAIN' in col_upper:
                        mountain_prop = float(morpho_grid[col].iloc[0])

                # Soil data
                non_rock_prop = 0.0
                rock_prop = 0.0

                for col in soil_grid.columns:
                    col_upper = str(col).upper()
                    if 'NON_ROCK' in col_upper or 'NONROCK' in col_upper:
                        non_rock_prop = float(soil_grid[col].iloc[0])
                    elif 'ROCK' in col_upper and 'NON' not in col_upper:
                        rock_prop = float(soil_grid[col].iloc[0])

                # Anthropization data
                non_anthro_prop = 0.0
                anthro_prop = 0.0

                for col in anthro_grid.columns:
                    col_upper = str(col).upper()
                    if 'NON_ANTHROPISED' in col_upper or 'NON_ANTHRO' in col_upper:
                        non_anthro_prop = float(anthro_grid[col].iloc[0])
                    elif 'ANTHROPISED' in col_upper and 'NON' not in col_upper:
                        anthro_prop = float(anthro_grid[col].iloc[0])

                print(f"      🔍 Grid {grid_id}: morpho=({plain_prop:.2f},{hill_prop:.2f},{mountain_prop:.2f}), "
                      f"soil=({non_rock_prop:.2f},{rock_prop:.2f}), anthro=({non_anthro_prop:.2f},{anthro_prop:.2f})")

                # Calculate incremental cost factors for this grid
                morpho_incremental_factor = (
                        plain_prop * cost_factors['k_morpho_plain'] +
                        hill_prop * cost_factors['k_morpho_hill'] +
                        mountain_prop * cost_factors['k_morpho_mountain']
                )

                soil_incremental_factor = (
                        non_rock_prop * cost_factors['k_soil_non_rock'] +
                        rock_prop * cost_factors['k_soil_rock']
                )

                anthro_incremental_factor = (
                        non_anthro_prop * cost_factors['k_anthro_non_anthropised'] +
                        anthro_prop * cost_factors['k_anthro_anthropised']
                )

                # Weight by intersection proportion and add to totals
                total_weighted_morpho_factor += morpho_incremental_factor * intersection_prop
                total_weighted_soil_factor += soil_incremental_factor * intersection_prop
                total_weighted_anthro_factor += anthro_incremental_factor * intersection_prop

                print(f"      🔍 Grid {grid_id} factors: morpho={morpho_incremental_factor:.3f}, "
                      f"soil={soil_incremental_factor:.3f}, anthro={anthro_incremental_factor:.3f}")

            except Exception as e:
                print(f"      ❌ Error processing grid {grid_id}: {e}")
                continue

        # Calculate total incremental geographical factor
        # Since intersection_proportions should sum to 1, these are already weighted averages
        total_incremental_geo_factor = (
                total_weighted_morpho_factor +
                total_weighted_soil_factor +
                total_weighted_anthro_factor
        )

        # Apply reasonable bounds and round to 3 decimal places
        total_incremental_geo_factor = max(total_incremental_geo_factor, -0.500)  # Max 50% cost reduction
        total_incremental_geo_factor = min(total_incremental_geo_factor, 2.000)  # Max 200% cost increase
        total_incremental_geo_factor = round(total_incremental_geo_factor, 3)

        print(f"      ✅ Processed {grids_processed} grids, total geo factor: {total_incremental_geo_factor:.3f}")

        # Provide feedback if no grids were processed
        if grids_processed == 0:
            print(f"      ⚠️  No geographical grids could be matched for pipeline DN {pipeline_DN:.0f}")

        return total_incremental_geo_factor

    def _set_options(self, options: dict):
        """Sets all provided options"""
        super()._set_options(options)

        try:
            self.options["length_km"] = round(float(options["length_km"]), 3)
        except KeyError:
            raise KeyError("You need to at least specify the pipeline length (length_km)")

        # Set options
        self._set_option_value("source", options)
        self.options["discount_rate"] = self.discount_rate

        if self.options["source"] == "Oeuvray":
            # Input units
            self.currency_in = "EUR"
            self.financial_year_in = 2024

            # Options with 3 decimal places for numerical values
            for o in self.default_options.keys():
                self._set_option_value(o, options)
                if isinstance(self.options[o], (int, float)):
                    if o == "massflow_evaluation_points":
                        # Ensure this remains an integer
                        self.options[o] = int(round(float(self.options[o])))
                    else:
                        self.options[o] = round(float(self.options[o]), 3)

            # Set geographical data options (required but can be empty)
            self.options["morpho_data"] = options.get("morpho_data", pd.DataFrame())
            self.options["soil_data"] = options.get("soil_data", pd.DataFrame())
            self.options["anthro_data"] = options.get("anthro_data", pd.DataFrame())
            self.options["intersected_grids"] = options.get("intersected_grids", [])
            self.options["intersected_proportions"] = options.get("intersected_proportions", [])
        else:
            raise ValueError("This source is not available")

    def calculate_indicators(self, options: dict):
        """
        Calculates financial indicators with incremental geographical factors
        """
        super().calculate_indicators(options)

        if self.options["source"] == "Oeuvray":
            # Import CO2Chain_Oeuvray here to avoid circular import issues
            from adopt_net0.database.components.networks.utilities.co2_pipelines_oeuvray import CO2Chain_Oeuvray

            if (self.options["massflow_min_kg_per_s"] == self.options["massflow_max_kg_per_s"]):
                range_massflow_kg_per_s = [self.options["massflow_min_kg_per_s"]]
            else:
                range_massflow_kg_per_s = np.linspace(
                    self.options["massflow_min_kg_per_s"],
                    self.options["massflow_max_kg_per_s"],
                    self.options["massflow_evaluation_points"],
                )

            calculation_module = CO2Chain_Oeuvray()
            self.financial_indicators["lifetime"] = calculation_module.universal_data["z_pumpcomp"]

            # Calculate costs for different mass flow rates
            costs = pd.DataFrame()
            geo_factors = pd.DataFrame()

            # Check if geographical data is available and valid
            geo_data_available = all([
                self.options.get("morpho_data") is not None,
                self.options.get("soil_data") is not None,
                self.options.get("anthro_data") is not None,
                self.options.get("intersected_grids") is not None,
                self.options.get("intersected_proportions") is not None,
                not self.options.get("morpho_data").empty if hasattr(self.options.get("morpho_data", None),
                                                                     'empty') else False,
                not self.options.get("soil_data").empty if hasattr(self.options.get("soil_data", None),
                                                                   'empty') else False,
                not self.options.get("anthro_data").empty if hasattr(self.options.get("anthro_data", None),
                                                                     'empty') else False,
                len(self.options.get("intersected_grids", [])) > 0,
                len(self.options.get("intersected_proportions", [])) > 0
            ])

            print(f"   🔍 Geographical data available: {geo_data_available}")

            for massflow_kg_per_s in range_massflow_kg_per_s:
                massflow_t_per_h = massflow_kg_per_s / 1000 * 3600
                self.options["massflow_kg_per_s"] = massflow_kg_per_s
                cost = calculation_module.calculate_cost(self.options)

                # Calculate incremental geographical factor if data is provided
                incremental_geo_factor = 0.000  # Default: no adjustment
                if geo_data_available:
                    try:
                        incremental_geo_factor = self._calculate_incremental_geo_factor(
                            massflow_kg_per_s,
                            self.options["intersected_grids"],
                            self.options["intersected_proportions"],
                            self.options["morpho_data"],
                            self.options["soil_data"],
                            self.options["anthro_data"]
                        )
                    except Exception as e:
                        print(f"      ❌ Error calculating geo factor: {e}")
                        incremental_geo_factor = 0.000

                # Correct for compression lifetime
                cr_pipeline = (
                        self.discount_rate * (1 + self.discount_rate) ** calculation_module.universal_data["z_pipe"]
                        / ((1 + self.discount_rate) ** calculation_module.universal_data["z_pipe"] - 1)
                )
                cr_compressor = (
                        self.discount_rate * (1 + self.discount_rate) ** calculation_module.universal_data["z_pumpcomp"]
                        / ((1 + self.discount_rate) ** calculation_module.universal_data["z_pumpcomp"] - 1)
                )
                correction_factor = cr_pipeline / cr_compressor

                costs.loc[massflow_t_per_h, "capex_pipeline"] = round(
                    cost["cost_pipeline"]["unit_capex"] * correction_factor, 3
                )
                costs.loc[massflow_t_per_h, "capex_compression"] = round(cost["cost_compression"]["unit_capex"], 3)
                costs.loc[massflow_t_per_h, "capex_total"] = round(
                    cost["cost_pipeline"]["unit_capex"] * correction_factor
                    + cost["cost_compression"]["unit_capex"], 3
                )

                # Store incremental geographical factor
                costs.loc[massflow_t_per_h, "incremental_geo_factor"] = round(incremental_geo_factor, 3)

                # Apply incremental geographical factor to total CAPEX
                original_capex = costs.loc[massflow_t_per_h, "capex_total"]
                updated_capex = round(original_capex * (1 + incremental_geo_factor), 3)
                costs.loc[massflow_t_per_h, "updated_capex_total"] = updated_capex

                costs.loc[massflow_t_per_h, "opex_var"] = round(
                    cost["cost_pipeline"]["opex_var"] * correction_factor
                    + cost["cost_compression"]["opex_var"], 3
                )
                costs.loc[massflow_t_per_h, "opex_fix"] = round(
                    (cost["cost_pipeline"]["opex_fix_abs"] + cost["cost_compression"]["opex_fix_abs"])
                    / costs.loc[massflow_t_per_h, "updated_capex_total"], 3
                )

                costs.loc[massflow_t_per_h, "specific_compression_energy"] = round(
                    cost["energy_requirements"]["specific_compression_energy"], 3
                )
                costs.loc[massflow_t_per_h, "levelized_cost"] = round(cost["levelized_cost"], 3)

                # Store geographical factor for reference
                geo_factors.loc[massflow_t_per_h, "incremental_geo_factor"] = round(incremental_geo_factor, 3)

            # Fit linear cost function to results using updated CAPEX
            costs["intercept"] = 1
            d = costs.reset_index(names="massflow_t_per_h")
            x = d[["massflow_t_per_h", "intercept"]]
            y = d["updated_capex_total"]

            linmodel = sm.OLS(y, x)
            linfit = linmodel.fit()
            coeff = linfit.params

            self.financial_indicators["gamma1"] = round(convert_currency(
                coeff["intercept"], self.financial_year_in, self.financial_year_out, self.currency_in,
                self.currency_out,
            ), 3)
            self.financial_indicators["gamma2"] = round(convert_currency(
                coeff["massflow_t_per_h"], self.financial_year_in, self.financial_year_out, self.currency_in,
                self.currency_out,
            ), 3)
            self.financial_indicators["gamma3"] = 0.000
            self.financial_indicators["gamma4"] = 0.000
            self.financial_indicators["opex_fixed"] = round(costs["opex_fix"].mean(), 3)
            self.financial_indicators["opex_variable"] = round(convert_currency(
                costs["opex_var"].mean(), self.financial_year_in, self.financial_year_out, self.currency_in,
                self.currency_out,
            ), 3)
            self.financial_indicators["levelized_cost"] = round(convert_currency(
                costs["levelized_cost"].mean(), self.financial_year_in, self.financial_year_out, self.currency_in,
                self.currency_out,
            ), 3)

            # Store geographical factors for reference
            self.geo_factors = geo_factors

            self.technical_indicators["energyconsumption"] = {
                "CO2captured": {"cons_model": 1, "k_flow": 0.000, "k_flowDistance": 0.000},
                "electricity": {
                    "cons_model": 1,
                    "k_flow": round(costs["specific_compression_energy"].mean(), 3),
                    "k_flowDistance": 0.000,
                },
            }

            # Write to json template
            self.json_data["Economics"]["gamma1"] = self.financial_indicators["gamma1"]
            self.json_data["Economics"]["gamma2"] = self.financial_indicators["gamma2"]
            self.json_data["Economics"]["gamma3"] = self.financial_indicators["gamma3"]
            self.json_data["Economics"]["gamma4"] = self.financial_indicators["gamma4"]
            self.json_data["Economics"]["OPEX_variable"] = self.financial_indicators["opex_variable"]
            self.json_data["Economics"]["opex_fixed"] = self.financial_indicators["opex_fixed"]
            self.json_data["Economics"]["lifetime"] = self.financial_indicators["lifetime"]
            self.json_data["Performance"]["energyconsumption"] = self.technical_indicators["energyconsumption"]

            return {
                "financial_indicators": self.financial_indicators,
                "technical_indicators": self.technical_indicators,
                "costs_detailed": costs,
                "geo_factors": geo_factors
            }