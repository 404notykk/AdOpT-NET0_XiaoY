from cmath import phase

import pandas as pd
from pathlib import Path
import numpy as np
import math


class CO2Transport_Oeuvray:
    """
    Calculates cost CO2 transport cost onshore or offshore, based on distance and inlet/outloet pressure.

    Minimizes the levelized cost of CO2 transport taking into account also electricity cost for compression

    Algorithm is based on Pauline Oeuvray, Johannes Burger, Simon Roussanaly, Marco Mazzotti, Viola Becattini (2024):
    Multi-criteria assessment of inland and offshore carbon dioxide transport options, Journal of Cleaner Production,
    """

    def __init__(self, fluid_properties_file="CO2IsothermalProperties.xlsx"):
        super().__init__()
        input_path = Path(__file__).parent.parent.parent.parent / Path(
            "./data/networks/co2_transport_oeuvray/"
        )
        if fluid_properties_file == "CO2IsothermalProperties.xlsx":
            print(f"Fluid property input path: {fluid_properties_file}")
            self.flue_gas = "Pure CO2"
        if fluid_properties_file == "CO2IsothermalProperties_0.05N2.xlsx":
            print(f"Fluid property input path: {fluid_properties_file}")
            self.flue_gas = "5% N2"
        if fluid_properties_file == "CO2IsothermalProperties_CPU1.xlsx":
            print(f"Fluid property input path: {fluid_properties_file}")
            self.flue_gas = "CPU1"
        if fluid_properties_file == "CO2IsothermalProperties_CPU2.xlsx":
            print(f"Fluid property input path: {fluid_properties_file}")
            self.flue_gas = "CPU2"
        if fluid_properties_file == "CO2IsothermalProperties_CPU3.xlsx":
            print(f"Fluid property input path: {fluid_properties_file}")
            self.flue_gas = "CPU3"
        if fluid_properties_file == "CO2IsothermalProperties_CPU4.xlsx":
            print(f"Fluid property input path: {fluid_properties_file}")
            self.flue_gas = "CPU4"

        fluid_properties_input_path = input_path / fluid_properties_file
        universal_data_input_path = input_path / "OtherData.xlsx"
        # input data
        self.CO2_m_kg_per_s = None
        self.p_outlet_system_mpa = None
        self.length_km = None
        self.timeframe = None
        self.total_m_kg_per_s = None
        self.force_phase = None
        self.phase = None
        self.electricity_price_eur_per_mw = None
        self.operating_hours_per_a = None
        self.p_initial_mpa = None
        self.terrain = None

        # results
        self.current_best_results = {}
        self.optimal_configuration = None

        # Fluid Properties
        self.fluid_properties = {}
        self.fluid_properties["277K"] = pd.read_excel(
            fluid_properties_input_path, "277K"
        ).set_index("Pressure (MPa)")
        self.fluid_properties["288K"] = pd.read_excel(
            fluid_properties_input_path, "288K"
        ).set_index("Pressure (MPa)")
        self.fluid_properties["303K"] = pd.read_excel(
            fluid_properties_input_path, "303K"
        ).set_index("Pressure (MPa)")
        self.fluid_properties["Constants"] = pd.read_excel(
            fluid_properties_input_path, "Constants"
        )

        # Universal data
        self.universal_data = pd.read_excel(
            universal_data_input_path, "Universal", index_col=0
        )["Value"].to_dict()

        # Fluid Specific Constants
        self.constants = pd.read_excel(
            fluid_properties_input_path, "Constants", index_col=0
        )["Value"].to_dict()

        # Terrain specific data
        self.terrain_specific_data = {}
        self.terrain_specific_data["gas"] = pd.read_excel(
            universal_data_input_path, "Terrain_specific_gas", index_col=0
        ).to_dict()
        self.terrain_specific_data["liquid"] = pd.read_excel(
            universal_data_input_path, "Terrain_specific_liquid", index_col=0
        ).to_dict()
        self.terrain_specific_data["gas"]["Offshore"]["OD_NPS"] = (
            pd.read_excel(universal_data_input_path, "OD_NPS", index_col=0, header=None)
            .loc["Offshore"]
            .to_numpy()
        )
        self.terrain_specific_data["gas"]["Onshore"]["OD_NPS"] = (
            pd.read_excel(universal_data_input_path, "OD_NPS", index_col=0, header=None)
            .loc["Onshore"]
            .to_numpy()
        )
        self.terrain_specific_data["liquid"]["Offshore"]["OD_NPS"] = (
            pd.read_excel(universal_data_input_path, "OD_NPS", index_col=0, header=None)
            .loc["Offshore"]
            .to_numpy()
        )
        self.terrain_specific_data["liquid"]["Onshore"]["OD_NPS"] = (
            pd.read_excel(universal_data_input_path, "OD_NPS", index_col=0, header=None)
            .loc["Onshore"]
            .to_numpy()
        )

        # Steel data
        self.steel_data = pd.read_excel(
            universal_data_input_path, "Steel_data", index_col=0
        )

        # Cost parameters
        self.cost_compressors = pd.read_excel(
            universal_data_input_path, "Compressor_costs", index_col=0
        )["Value"].to_dict()
        self.cost_pumps = pd.read_excel(
            universal_data_input_path, "Pump_costs", index_col=0
        )["Value"].to_dict()

        # Financial indicators
        self.lifetime = None
        self.discount_rate = None
        self.unit_capex = None
        self.opex_fix = None
        self.opex_var = None

    def _preprocess_data(self):
        """
        Preprocesses data based on input
        """
        #Critical Pressure
        self.pcrit = self.constants["Pcrit"]
        print(f"pcrit: {self.pcrit}")

        #Molar Mass
        self.M = self.constants["M_kg_per_mol"]
        print(f"M: {self.M}")

        # Kappa
        self.kappa = self.constants["kappa"]
        print(f"kappa: {self.kappa}")

        # densities and viscosities with phase-aware interpolation
        # liquid
        liq_off = self._get_density_viscosity(terrain="Offshore", phase="liquid")  # default 8 MPa
        liq_on = self._get_density_viscosity(terrain="Onshore", phase="liquid")  # default 8 MPa
        self.terrain_specific_data["liquid"]["Offshore"]["rho_kg_per_m3"] = liq_off["density"]
        self.terrain_specific_data["liquid"]["Offshore"]["mu_Pas"] = liq_off["viscosity"]
        self.terrain_specific_data["liquid"]["Onshore"]["rho_kg_per_m3"] = liq_on["density"]
        self.terrain_specific_data["liquid"]["Onshore"]["mu_Pas"] = liq_on["viscosity"]

        # gas
        gas_off = self._get_density_viscosity(terrain="Offshore", phase="gas")  # default 1.5 MPa
        gas_on = self._get_density_viscosity(terrain="Onshore", phase="gas")  # default 1.5 MPa
        self.terrain_specific_data["gas"]["Offshore"]["rho_kg_per_m3"] = gas_off["density"]
        self.terrain_specific_data["gas"]["Offshore"]["mu_Pas"] = gas_off["viscosity"]
        self.terrain_specific_data["gas"]["Onshore"]["rho_kg_per_m3"] = gas_on["density"]
        self.terrain_specific_data["gas"]["Onshore"]["mu_Pas"] = gas_on["viscosity"]

        # steel grade availability by timeframe
        if self.timeframe == "near-term":
            self.steel_data = self.steel_data[self.steel_data["available_near_term"] == 1]
        elif self.timeframe == "mid-term":
            self.steel_data = self.steel_data[self.steel_data["available_mid_term"] == 1]
        elif self.timeframe == "long-term":
            self.steel_data = self.steel_data[self.steel_data["available_long_term"] == 1]
        else:
            raise ValueError("Time frame not available")

    def _get_density_viscosity(self, terrain: str, phase: str, pressure_mpa: float | None = None) -> dict:
        """
        Returns {"density": kg/m3, "viscosity": Pa*s} at the requested pressure.
        Uses only rows with the requested Phase from the correct sheet.
        Defaults: 1.5 MPa for gas and 8.0 MPa for liquid.
        """
        phase = phase.lower().strip()
        if phase not in {"gas", "liquid"}:
            raise ValueError("phase must be 'gas' or 'liquid'")

        if pressure_mpa is None:
            pressure_mpa = 1.5 if phase == "gas" else self.constants["Pcrit"] + 0.5

        if terrain == "Offshore":
            sheet_key = "277K"
        elif terrain == "Onshore":
            sheet_key = "288K"
        else:
            raise ValueError("terrain must be 'Offshore' or 'Onshore'")

        df = self.fluid_properties[sheet_key]

        required_cols = {"Density (kg/m3)", "Viscosity (Pa*s)", "Phase"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns in sheet {sheet_key}: {missing}; flue_gas: {self.flue_gas}")

        # filter by phase label and prepare numeric pressure axis
        sub = df[df["Phase"].astype(str).str.lower() == phase].copy()
        if sub.empty:
            raise ValueError(f"No rows with Phase == '{phase}' in sheet {sheet_key}")

        sub["_p_"] = pd.to_numeric(sub.index, errors="coerce")
        sub = sub.dropna(subset=["_p_"]).sort_values("_p_")
        sub = sub[~sub["_p_"].duplicated(keep="first")]

        x = sub["_p_"].to_numpy()
        if len(x) == 0:
            raise ValueError(f"No valid numeric pressures in sheet {sheet_key} for phase {phase}")

        def interp_col(col_name: str) -> float:
            y = pd.to_numeric(sub[col_name], errors="coerce").to_numpy()
            if len(y) == 0:
                raise ValueError(f"No numeric data in column '{col_name}'")
            if len(x) == 1:
                return float(y[0])
            xp = float(pressure_mpa)
            if xp <= x[0]:
                x0, x1 = x[0], x[1]
                y0, y1 = y[0], y[1]
                return float(y0 + (y1 - y0) * (xp - x0) / (x1 - x0))
            if xp >= x[-1]:
                x0, x1 = x[-2], x[-1]
                y0, y1 = y[-2], y[-1]
                return float(y0 + (y1 - y0) * (xp - x0) / (x1 - x0))
            return float(np.interp(pressure_mpa, x, y))

        density = interp_col("Density (kg/m3)")
        viscosity = interp_col("Viscosity (Pa*s)")
        return {"density": density, "viscosity": viscosity}

    def _calculate_pipeline_configuration(
        self, pinlet_mpa, poutlet_mpa, id_calc_m, delta_p_inlet
    ):

        # Get data
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        # Calculate v, re, f
        id_calc_m_initial= id_calc_m
        v_m_per_s = self._calculate_velocity(id_calc_m)
        re = self._calculate_reynolds(id_calc_m, v_m_per_s)
        f = self._calculate_darcyweisbach(id_calc_m, re)
        while pinlet_mpa <= terrain_data["PinletMAX_MPa"]:
            n_pump = 0
            while n_pump <= terrain_data["Npump_max"]:
                print(f"-----------------------------")
                print(f"Calculating {round(pinlet_mpa,2)} MPA inlet pressure with {n_pump} Pumps")

                delta_p_design_pa_per_m = self._calculate_design_pressure_drop(
                    pinlet_mpa, poutlet_mpa, n_pump
                )
                if delta_p_design_pa_per_m <= 0:  # delta_p_design_pa_per_m can be due to elevation changes <0; those cases are not considered
                    n_pump = n_pump + 1
                    if n_pump > terrain_data["Npump_max"]:
                        break
                    continue


                id_calc_m_new = id_calc_m_initial
                counter = 0
                while True:
                    counter += 1
                    id_calc_m_old = id_calc_m_new
                    l_pump_m = self._calculate_max_distance_pumps(pinlet_mpa, poutlet_mpa, delta_p_design_pa_per_m, n_pump)

                    id_calc_m_new = self._calculate_inner_diameter(
                        pinlet_mpa, poutlet_mpa, f, l_pump_m, delta_p_design_pa_per_m
                    )

                    v_m_per_s = self._calculate_velocity(id_calc_m_new)
                    re = self._calculate_reynolds(id_calc_m_new, v_m_per_s)
                    f = self._calculate_darcyweisbach(id_calc_m_new, re)

                    if abs(id_calc_m_new-id_calc_m_old) < 0.01 or counter > 10:
                        id_calc_m = id_calc_m_new
                        break
                if (
                    max(
                        terrain_data["OD_NPS"]
                        - 2 * terrain_data["dtRatio"] * terrain_data["OD_NPS"]
                        - id_calc_m
                    )
                    > 0
                ):
                    best_steel_grade = self._find_best_steel_grade(id_calc_m, pinlet_mpa)
                    if best_steel_grade is None:
                        print(f"ISSUE: No suitable steel grade found, skipping configuration {round(pinlet_mpa,2)} MPA inlet pressure with {n_pump} Pumps")
                        n_pump = n_pump + 1
                        if n_pump > terrain_data["Npump_max"]:
                            break
                        continue
                    print(f"best_steel_grade: {best_steel_grade.index[0]}")
                    current_result = self._levelized_cost(
                        best_steel_grade, pinlet_mpa, poutlet_mpa, n_pump
                    )

                    if current_result is None:
                        n_pump = n_pump + 1
                        print(f"ISSUE: No suitable solution found, skipping configuration {round(pinlet_mpa,2)} MPA inlet pressure with {n_pump} Pumps")
                        if n_pump > terrain_data["Npump_max"]:
                            break
                        continue

                    if current_result["lc"] < self.current_best_results["lc"]:
                        self.current_best_results = current_result
                        print(
                            f"New best config found with steel grade {best_steel_grade.index[0]} with {current_result['n_pumps']} Pump(s) at {pinlet_mpa} MPA inlet pressure. LC: {current_result['lc']}"
                        )
                else:
                    print(f"ISSUE: Max OD_NPS was not enough")

                if self.terrain == "Onshore":
                    n_pump = n_pump + 1
                else: break
            pinlet_mpa = pinlet_mpa + delta_p_inlet
            print(f"p_inlet increased")

        return self.current_best_results

    def _find_best_steel_grade(self, id_calc_m, pinlet_mpa):
        """
        Calculates min cost for different steel grades

        :param float id_calc_m: starting inner diameter of pipe in m
        :param float p_inlet_mpa: inlet pressure in MPa
        :param float p_outlet_mpa: outlet pressure in MPa
        :return: cost factors for different steel grades
        :rtype: pd.DataFrame
        """

        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        # Max operating pressure
        max_p_mpa = self._calculate_max_operating_pressure(pinlet_mpa)
        print(f": Steel grade calculation --------------------------------")
        pipe_cost_for_different_steel_grades = []
        id_calc_m_initial = id_calc_m

        for idx, steel_grade in self.steel_data.iterrows():
            # Starting value
            id_nps_m = 0
            id_calc_m = id_calc_m_initial

            od_nps_chosen = None
            for od in sorted(terrain_data["OD_NPS"]):
                t_m = self._calculate_pipe_thickness(od, max_p_mpa, steel_grade.S_MPa)
                id_nps_m = od - 2 * t_m
                if id_nps_m >= id_calc_m:
                    od_nps_chosen = od
                    break

            if od_nps_chosen is None:
                print(f"Warning: No suitable OD found for required inner diameter for steel grade {idx}!")
                continue

            pipe_cost = self._calculate_pipe_costs(
                t_m, od_nps_chosen, steel_grade.Csteel_EUR_per_kg
            )
            pipe_cost["id_nps_m"] = id_nps_m
            pipe_cost["t_m"] = t_m
            pipe_cost["od_nps_m"] = od_nps_chosen
            pipe_cost["steel_grade"] = idx

            pipe_cost_for_different_steel_grades.append(pipe_cost)

        if not pipe_cost_for_different_steel_grades:
            print("Error: No steel grade provides a suitable OD for required inner diameter! Stopping calculation.")
            return None

        pipe_costs = pd.DataFrame(pipe_cost_for_different_steel_grades)
        pipe_costs = pipe_costs.set_index("steel_grade")

        return pipe_costs[pipe_costs["capex_total"] == pipe_costs["capex_total"].min()]

    def _calculate_velocity(self, id_calc_m):
        """
        Calculates velocity through a pipeline

        :param float id_calc_m: inner diameter in m
        :return: flow rate in m/s
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        return (
            4
            * self.total_m_kg_per_s
            / (id_calc_m**2 * math.pi * terrain_data["rho_kg_per_m3"])
        )

    def _calculate_reynolds(self, IDNPS_m, v_m_per_s):
        """
        Calculates reynolds number

        :param float IDNPS_m: inner diameter in m
        :param float v_m_per_s: flow rate in m/s
        :return: reynolds number
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        return (
            terrain_data["rho_kg_per_m3"] * IDNPS_m * v_m_per_s / terrain_data["mu_Pas"]
        )

    def _calculate_darcyweisbach(self, IDNPS_m, Re):
        """
        Calculates darcy-weisbach friction factor

        :param float IDNPS_m: inner diameter in m
        :param float Re: reynolds number
        :return: friction factor
        :rtype: float
        """
        return (
            1
            / (
                -1.8
                * math.log10(
                    (self.universal_data["epsilon_m"] / IDNPS_m / 3.7) ** 1.11
                    + 6.9 / Re
                )
            )
        ) ** 2

    def _calculate_total_massflow(self, m_co2, flue_gas="Pure CO2"):

        if flue_gas == "Pure CO2" :
            total_massflow = m_co2
            print(f"Calculation executed for pure CO₂ | "
                  f"CO₂ mass flow: {m_co2:.2f} kg/s | Total mass flow: {total_massflow:.2f} kg/s")
            return total_massflow
        elif flue_gas == "5% N2":
            total_massflow = m_co2 / 0.95
            print(f"Calculation executed for impure CO₂, with 5 wt% N2 | CO₂ mass flow: {m_co2:.2f} kg/s | Total mass flow: {total_massflow:.2f} kg/s")
            return total_massflow
        elif flue_gas == "CPU1":
            total_massflow = m_co2 / 0.829
            print(f"Calculation executed for CPU1 with impurities: Ar: 3.1 wt%, N2: 7.8 wt%, O2: 6.2 wt% | CO₂ mass flow: {m_co2:.2f} kg/s | Total mass flow: {total_massflow:.2f} kg/s")
            return total_massflow
        elif flue_gas == "CPU2":
            total_massflow = m_co2 / 0.975
            print(f"Calculation executed for CPU2 with impurities: Ar: 0.5 wt%, N2: 1.0 wt%, O2: 1.0 wt% | CO₂ mass flow: {m_co2:.2f} kg/s | Total mass flow: {total_massflow:.2f} kg/s")
            return total_massflow
        elif flue_gas == "CPU3":
            total_massflow = m_co2 / 0.925
            print(f"Calculation executed for CPU3 with impurities: Ar: 1.4 wt%, N2: 3.2 wt%, O2: 2.9 wt% | CO₂ mass flow: {m_co2:.2f} kg/s | Total mass flow: {total_massflow:.2f} kg/s")
            return total_massflow
        elif flue_gas == "CPU4":
            total_massflow = m_co2
            print(f"Calculation executed for CPU4, pure CO₂ | CO₂ mass flow: {m_co2:.2f} kg/s | Total mass flow: {total_massflow:.2f} kg/s")
            return total_massflow

    def _calculate_compressor_outlet_gas(
        self, l_pump_m, p_ave_mpa, poutlet_mpa, id_nps_m
    ):
        """
        Caclulates actual outlet pressure of the compressor for gas transport in Pa

        :param float f: Darcy-Weisbach friction factor
        :param float l_pump_m: lmax distance between pumps in m
        :param float pinlet_mpa: pressure at inlet in MPa
        :param float poutlet_mpa: pressure at outlet in MPa
        :return: Actual outlet pressure of the compressor in Pa
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        v_m_per_s = self._calculate_velocity(id_nps_m)
        re = self._calculate_reynolds(id_nps_m, v_m_per_s)
        f = self._calculate_darcyweisbach(id_nps_m, re)

        p_ave_pa = p_ave_mpa * 1e6
        z_ave = self._calculate_compressibility_factor(p_ave_pa, terrain_data["T_degC"])
        t_ave_k = terrain_data["T_degC"] + 273.15

        return (
            16
            * z_ave
            * self.universal_data["R_J_per_mol_per_K"]
            * t_ave_k
            * self.total_m_kg_per_s**2
            * f
            * l_pump_m
            / (math.pi**2.0 * id_nps_m**5.0 * self.constants["M_kg_per_mol"])
            + 2
            * self.universal_data["g_m_per_s2"]
            * p_ave_pa**2.0
            * self.constants["M_kg_per_mol"]
            * self.universal_data["z_m"]
            / (z_ave * t_ave_k * self.universal_data["R_J_per_mol_per_K"])
            + (poutlet_mpa * 1e6) ** 2
        ) ** 0.5

    def _calculate_pressure_last_pump(
        self, l_pump_m, n_pumps, delta_p_act_pa_m
    ):
        """
        Calculates outlet pressure of the last pump

        :param float p_outlet_system_mpa: pressure at outlet in MPa
        :param float l_pump_m: lmax distance between pumps in m
        :param float n_pump: number of pumps
        :param float delta_p_act_pa_m: actual pressure drop in Pa/m
        :return: outlet pressure of last pump in pa
        :rtype: float
        """
        return (
            self.p_outlet_system_mpa * 1e6
            + (self.length_km * 1000 - l_pump_m * n_pumps) * delta_p_act_pa_m
        )

    def _calculate_design_pressure_drop(self, pinlet_mpa, poutlet_mpa, n_pump):
        """
        Calculates design pressure drop

        :param float pinlet_mpa: pressure at inlet in MPa
        :param float poutlet_mpa: pressure at outlet
        :param float n_pump: number of pumps
        :return: pressure drop in Pa
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        pinlet_pa = pinlet_mpa * 1e6
        poutlet_pa = poutlet_mpa * 1e6
        l_m = self.length_km * 1e3
        return (
            (pinlet_pa - poutlet_pa) * (n_pump + 1)
            + self.universal_data["g_m_per_s2"]
            * terrain_data["rho_kg_per_m3"]
            * self.universal_data["z_m"]
        ) / l_m

    def _calculate_recompression_energy(self, pinlet_mpa, poutlet_mpa):
        if self.phase == "gas":
            return self._calculate_recompression_energy_gas(pinlet_mpa, poutlet_mpa)
        else:
            return self._calculate_recompression_energy_liquid(pinlet_mpa, poutlet_mpa)

    def _calculate_recompression_energy_liquid(self, pinlet_mpa, poutlet_mpa):
        """
        Calculates specific energy of a gas compressor

        :param p_out_pa: pressure at outlet in Pa
        :param p_in_pa: pressure at inlet in Pa
        :return: compression energy in MJ/kg
        :rtype: float
        """

        p_pump_outlet = pinlet_mpa
        p_pump_inlet = poutlet_mpa
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        return (p_pump_outlet - p_pump_inlet) / (
            self.universal_data["etaPump"] * terrain_data["rho_kg_per_m3"]
        )


    def _calculate_recompression_energy_gas(self, poutlet_mpa, pinlet_mpa):
        """
        Calculates specific energy of a gas compressor

        :param p_out_pa: pressure at outlet in Pa
        :param p_in_pa: pressure at inlet in Pa
        :return: compression energy in MJ/kg
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        pr = self.universal_data["PR"]
        pinlet_pa = pinlet_mpa * 1e6
        poutlet_pa = poutlet_mpa * 1e6
        t_ave_c = terrain_data["T_degC"]

        # Onshore [15°C] / Offshore [4°C] compression
        #z = 0.910912883        #initially used compressibility factor
        t_comp_k = t_ave_c + 273.15
        z = self._calculate_compressibility_factor(pinlet_pa, t_ave_c)                    #z is calculated for inlet conditions, could be changed every stage

        # gas recompression
        n_stages = math.ceil(math.log(poutlet_pa / pinlet_pa) / math.log(pr))
        p1_pa = poutlet_pa
        pr = (p1_pa / pinlet_pa) ** (1 / n_stages)

        e_comp_J_per_kg = z * self.universal_data[
            "R_J_per_mol_per_K"
        ] * t_comp_k * n_stages * self.constants["kappa"] * (
            pr ** ((self.constants["kappa"] - 1) / self.constants["kappa"])
            - 1
        ) / (
            self.constants["M_kg_per_mol"]
            * self.universal_data["etaIso"]
            * self.universal_data["etaMech"]
            * (self.constants["kappa"] - 1)
        )

        return e_comp_J_per_kg / 1e6



    def _calculate_initial_compression_energy(
        self, poutlet_mpa, p_initial_mpa
    ):
        """
        Calculates specific energy of a gas compressor

        :param p_out_pa: pressure at outlet in Pa
        :param p_in_pa: pressure at inlet in Pa
        :return: compression energy in MJ/kg
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        pr = self.universal_data["PR"]
        p_comp_in_pa = p_initial_mpa * 1e6
        poutlet_pa = poutlet_mpa * 1e6

        #initial_compression
        #z = 0.994799474        #compressibility used in the first place - wrong; for 1.1 bar
        t_comp_k = 303.15
        t_ave_c = 30
        z = self._calculate_compressibility_factor(p_comp_in_pa, t_ave_c)

        if self.phase == "liquid":
            # liquid
            n_stages = math.ceil(math.log((7e6 + 5e5) / p_comp_in_pa) / math.log(pr))
            p1_pa = p_comp_in_pa * pr**n_stages
            if p_comp_in_pa > 7.5e6:
                n_stages = 0
                pr = 0
                p1_pa = p_comp_in_pa
            else:
                pr = (p1_pa / p_comp_in_pa) ** (1 / n_stages)
        else:
            # gas
            n_stages = math.ceil(math.log(poutlet_pa / p_comp_in_pa) / math.log(pr))
            p1_pa = poutlet_pa

            pr = (p1_pa / p_comp_in_pa) ** (1 / n_stages)


        #compressor part (gas)
        e_gas_J_per_kg = (
                z * self.universal_data["R_J_per_mol_per_K"] * t_comp_k
                * n_stages * self.constants["kappa"]
                * (pr ** ((self.constants["kappa"] - 1) / self.constants["kappa"]) - 1)
                / (self.constants["M_kg_per_mol"] * self.universal_data["etaIso"]
                   * self.universal_data["etaMech"] * (self.constants["kappa"] - 1))
        )
        #pump part liquid
        dp_pump = max(poutlet_pa - p1_pa, 0.0)
        e_pump_J_per_kg = dp_pump / (self.universal_data["etaPump"] * terrain_data["rho_kg_per_m3"])

        e_total_MJ_per_kg = (e_gas_J_per_kg + e_pump_J_per_kg) / 1e6
        return e_total_MJ_per_kg, e_gas_J_per_kg / 1e6, e_pump_J_per_kg / 1e6, dp_pump

    def _calculate_max_distance_pumps(
        self, pinlet_mpa, poutlet_mpa, delta_p_pa_per_m, n_pump):
        """
        Calculates distance between pumping stations in m

        :param float pinlet_mpa: pressure at inlet in MPa
        :param float poutlet_mpa: pressure at outlet in MPa
        :param float delta_p_pa_per_m: pressure drop in Pa
        :return: maximum distance between pumping stations in m
        :rtype: float
        """
        pinlet_pa = pinlet_mpa * 1e6
        poutlet_pa = poutlet_mpa * 1e6

        if self.terrain == "Onshore" and n_pump != 0:
            return (pinlet_pa - poutlet_pa) / delta_p_pa_per_m
        else:
            return self.length_km*1000

    def _calculate_max_distance_compressors(self, pinlet_mpa, poutlet_mpa, id_nps_m, n_pump):
        """
        Calculates required inner diameter for gaseous transport in m

        :param float pinlet_mpa: pressure at inlet in MPa
        :param float poutlet_mpa: pressure at outlet in MPa
        :param float f: Darcy-Weisbach friction factor
        :param float id_nps_m: inner diameter in m
        :return: lmax distance between recompresion stations for gaseous transport in m
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        pinlet_pa = pinlet_mpa * 1e6
        poutlet_pa = poutlet_mpa * 1e6
        t_ave_k = terrain_data["T_degC"] + 273.15
        p_ave_pa = self._calculate_average_pressure(poutlet_pa, pinlet_pa)
        z_ave = self._calculate_compressibility_factor(p_ave_pa, terrain_data["T_degC"])
        v_m_per_s = self._calculate_velocity(id_nps_m)
        re = self._calculate_reynolds(id_nps_m, v_m_per_s)
        f = self._calculate_darcyweisbach(id_nps_m, re)


        if self.terrain == "Onshore" and self.phase == "gas" and n_pump != 0:
            return (
            id_nps_m**5
            * math.pi**2
            * (
                self.constants["M_kg_per_mol"]
                * z_ave
                * t_ave_k
                * self.universal_data["R_J_per_mol_per_K"]
                * (poutlet_pa ** 2 - pinlet_pa ** 2)
                + 2
                * self.universal_data["g_m_per_s2"]
                * p_ave_pa**2
                * self.constants["M_kg_per_mol"] ** 2.0
                * self.universal_data["z_m"]
            )
            / (
            -16
            * z_ave**2
            * self.universal_data["R_J_per_mol_per_K"] ** 2
            * t_ave_k**2
            * self.total_m_kg_per_s**2
            * f
            )
            )
        else:
            return self.length_km*1000


    def _calculate_number_pumps(self, l_pump_m, terrain, n_pump):
        """
        Calculates number of pumps required

        :param float l_pump_m: maximum distance between pumping stations in m
        :return: number of pumping stations
        :rtype: float
        """

        if terrain == "Onshore" and n_pump != 0:
            return math.floor(self.length_km * 1000.0 / l_pump_m)
        else:
            return 0

    def _calculate_inner_diameter(
        self, pinlet_mpa, poutlet_mpa, f, l_pump_m, delta_p_design_pa_per_m
    ):
        """
        Calculates required inner diameter in m

        :param float pinlet_mpa: pressure at inlet in MPa
        :param float poutlet_mpa: pressure at outlet in MPa
        :param float f: Darcy-Weisbach friction factor
        :param float l_pump_m: lmax distance between pumps in m
        :return: required inner diameter for gaseous transport in m
        :rtype: float
        """
        if self.phase == "gas":
            return self._calculate_inner_diameter_gas(pinlet_mpa, poutlet_mpa, f, l_pump_m)
        else:
            return self._calculate_inner_diameter_liquid(f, delta_p_design_pa_per_m)

    def _calculate_inner_diameter_liquid(self, f, delta_p_design_pa_per_m):
        """
        Calculates required inner diameter for liquid transport in m

        :param float f: Darcy-Weisbach friction factor
        :param float delta_p_design_pa_per_m: design pressure drop in Pa/m]
        :return: required inner diameter for gaseous transport in m
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        return (
            8
            * f
            * self.total_m_kg_per_s**2
            / (math.pi**2 * terrain_data["rho_kg_per_m3"] * delta_p_design_pa_per_m)
        ) ** (1 / 5)

    def _calculate_inner_diameter_gas(self, pinlet_mpa, poutlet_mpa, f, l_pump_m):
        """
        Calculates required inner diameter for gaseous transport in m

        :param float pinlet_mpa: pressure at inlet in MPa
        :param float poutlet_mpa: pressure at outlet in MPa
        :param float f: Darcy-Weisbach friction factor
        :param float l_pump_m: lmax distance between pumps in m
        :return: required inner diameter for gaseous transport in m
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        pinlet_pa = pinlet_mpa * 1e6
        poutlet_pa = poutlet_mpa * 1e6
        t_ave_k = terrain_data["T_degC"] + 273.15

        p_ave_pa = self._calculate_average_pressure(poutlet_pa, pinlet_pa)
        z_ave = self._calculate_compressibility_factor(p_ave_pa, terrain_data["T_degC"])
        return (
            -16
            * z_ave**2
            * self.universal_data["R_J_per_mol_per_K"] ** 2
            * t_ave_k**2
            * self.total_m_kg_per_s**2
            * f
            * l_pump_m
            / (
                math.pi**2
                * (
                    self.constants["M_kg_per_mol"]
                    * z_ave
                    * t_ave_k
                    * self.universal_data["R_J_per_mol_per_K"]
                    * (poutlet_pa**2 - pinlet_pa**2)
                    + 2
                    * self.universal_data["g_m_per_s2"]
                    * p_ave_pa**2
                    * self.constants["M_kg_per_mol"] ** 2.0
                    * self.universal_data["z_m"]
                )
            )
        ) ** (1 / 5)

    def _calculate_pipe_thickness(self, od_nps_m, max_p_mpa, s_mpa):
        """
        Calculates required inner diameter for gaseous transport in m

        :param float od_nps_m: outer diameter of the nominal pipe size in m
        :param float max_p_mpa: maximum allowable operating pressure in MPa
        :param float s_mpa: minimum yield stress in MPa
        :param float e: longitudinal joint factor
        :return: pipe thickness in m
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        t_m = (
            od_nps_m
            * max_p_mpa
            / (2.0 * s_mpa * terrain_data["F"] * self.universal_data["E"])
            + self.universal_data["CA_m"]
        )
        if t_m / od_nps_m < terrain_data["dtRatio"]:
            t_m = od_nps_m * terrain_data["dtRatio"]

        return math.ceil(t_m * 2000) / 2000

    def _calculate_max_operating_pressure(self, p_inlet_mpa):
        """
        Calculates required inner diameter for gaseous transport in m

        :param float p_inlet_mpa: inlet pressure in MPa
        :return: maximum allowable operating pressure in MPa
        :rtype: float
        """
        return math.ceil(p_inlet_mpa * 1.1 * 10) / 10

    def _calculate_actual_pressure_drop(self, id_nps_m):
        """
        Calculates actual pressure drop

        :param float f: Darcy-Weisbach friction factor
        :param float id_nps_m: inner diameter of the nominal pipe size in m
        :return: pressure drop in Pa/m
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        v_m_per_s = self._calculate_velocity(id_nps_m)
        re = self._calculate_reynolds(id_nps_m, v_m_per_s)
        f = self._calculate_darcyweisbach(id_nps_m, re)

        return (
            8
            * f
            * self.total_m_kg_per_s**2
            / (math.pi**2 * terrain_data["rho_kg_per_m3"] * id_nps_m**5)
        )

    def _calculate_average_pressure(self, poutlet_pa, pinlet_pa):
        """
        Calculates average pressure in pa

        :param float pinlet_pa: pressure at inlet in Pa
        :param float poutlet_pa: pressure at outlet in Pa
        :return: average pressure in Pa
        :rtype: float
        """

        return (
            2
            * (
                poutlet_pa
                + pinlet_pa
                - poutlet_pa * pinlet_pa / (poutlet_pa + pinlet_pa)
            )
            / 3
        )

    def _calculate_compressibility_factor(self, p_ave_pa, t_ave_c):
        """
        Calculates average compressability factor
        change
        :param float p_ave_pa: average pressure in Pa
        :param float t_ave_c: average temperature in C
        :return: compressibility factor
        :rtype: float
        """
        if t_ave_c == 4:
            fluid_properties = self.fluid_properties["277K"]
        elif t_ave_c == 15:
            fluid_properties = self.fluid_properties["288K"]
        elif t_ave_c == 30:
            fluid_properties = self.fluid_properties["303K"]
        else:
            raise ValueError("Temperature is not allowed")

        return np.interp(
            p_ave_pa * 1e-6,
            fluid_properties.index,
            fluid_properties["Compressibility factor Z"],
        )

    def _calculate_pipe_costs(self, t_m, od_nps_m, c_steel_eur_per_kg):
        """
        Calculates pipeline costs

        :param float t_m: pipe thickness in m
        :param float od_nps_m: outer diameter in m
        :param float c_steel_eur_per_kg: cost of steel in eur/kg
        :return:
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        cost_factors = {}
        cost_factors["material"] = float(
            t_m
            * math.pi
            * (od_nps_m - t_m)
            * self.length_km
            * 1000
            * self.universal_data["rhoSteel_kg_per_m3"]
            * c_steel_eur_per_kg
            * self.universal_data["SteelFactor"]
        )
        cost_factors["labor"] = float(
            od_nps_m * self.length_km * 1000 * self.universal_data["Clab_EUR_per_m2"]
        )
        cost_factors["row"] = float(
            self.length_km * 1000 * terrain_data["CROW_EUR_per_m"]
        )
        cost_factors["misc"] = float(
            self.universal_data["mu_misc"]
            * (cost_factors["material"] + cost_factors["labor"])
        )
        cost_factors["capex_total"] = float(sum(cost_factors.values()))
        cost_factors["opex_fix"] = float(
            cost_factors["capex_total"] * self.universal_data["muOMpipe"]
        )

        return cost_factors

    def _calculate_compressor_cost(self, w_mw, phase):
        """
        Investment costs for pumps of compressors

        :param float w_mw: pump capacity in MW
        :return: pump cost in eur
        :rtype: float
        """
        if phase == "gas":
            return self._calculate_compressor_cost_gas(w_mw)
        else:
            return self._calculate_compressor_cost_liquid(w_mw)

    def _calculate_compressor_cost_liquid(self, w_mw):
        """
        Investment costs of pump in eur

        :param float w_mw: pump capacity in MW
        :return: pump cost in eur
        :rtype: float
        """
        n = math.ceil(w_mw / self.cost_pumps["WpumpMAX_MW"])
        cost = self.cost_pumps["Ipump0_EUR"] * ((w_mw * 1e3) ** 0.58) * n**0.32
        return cost

    def _calculate_compressor_cost_gas(self, w_mw):
        """
        Investment costs of compressor in eur

        :param float w_mw: compressor capacity in MW
        :return: compressor cost in eur
        :rtype: float
        """
        print(f"w_mw: {w_mw}")
        n = math.ceil(w_mw / self.cost_compressors["WcompMAX_MW"])
        if n == 0:
            w1 = 0
        else:
            w1 = w_mw / n

        cost = (
            self.cost_compressors["Icomp0_EUR"]
            * ((w1 / self.cost_compressors["Wcomp0_MW"]) ** self.cost_compressors["y"])
            * n ** self.cost_compressors["me"]
        )

        if cost < 0:
            cost = 0

        return cost

    def _calculate_recompression_energy_cost(self, w_comp_mw_total):
        """
        Calculates total recompression energy costs

        :param float w_comp_mw_total: total recompression capacity
        :return: total recompression energy cost
        :rtype: float
        """
        return (
            w_comp_mw_total
            * self.operating_hours_per_a
            * self.electricity_price_eur_per_mw
        )
    def _calculate_purification_cost(self):
        n = 0.8
        m = self.total_m_kg_per_s
        working_hours = self.operating_hours_per_a
        reference_total_m_t_per_h = 342.7
        reference_working_hours = 8460
        if self.flue_gas in {"Pure CO2", "5% N2"}:
            return 0, 0
        elif self.flue_gas == "CPU1":
            CAPEX_total_yearly = 33e6
            OPEX_total_yearly = 28e6
        elif self.flue_gas == "CPU2":
            CAPEX_total_yearly = 34e6
            OPEX_total_yearly = 32e6
        elif self.flue_gas == "CPU3":
            CAPEX_total_yearly = 42e6
            OPEX_total_yearly = 36e6
        elif self.flue_gas == "CPU4":
            CAPEX_total_yearly = 51e6
            OPEX_total_yearly = 34e6
        else:
            print(f"Error: Chosen flues gas {self.flue_gas} is not supported in purification cost calculation!")
            return 0, 0

        capex_purification = CAPEX_total_yearly * (m*working_hours*3.6/(reference_total_m_t_per_h*reference_working_hours))**n
        opex_purification = OPEX_total_yearly * (m*working_hours*3.6/(reference_total_m_t_per_h*reference_working_hours))
        return capex_purification, opex_purification

    def _calculate_levelized_cost(self, result):

        capex_purification, opex_purification = self._calculate_purification_cost()
        cr_purification = 0.1
        cr_pipe = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.universal_data["z_pipe"]
            / ((1 + self.discount_rate) ** self.universal_data["z_pipe"] - 1)
        )
        cr_pump_compressions = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.universal_data["z_pumpcomp"]
            / ((1 + self.discount_rate) ** self.universal_data["z_pumpcomp"] - 1)
        )
        levelized_costs = (
            cr_pipe * result["capex_pipe"]
            + cr_pump_compressions
            * (result["capex_recompression"] + result["capex_initial_compression"])
            + cr_purification * capex_purification
            + result["opex_pipe"]
            + result["opex_fix_compression"]                                            #OM Pump + Comp
            + result["opex_energy_recompression"]
            + result["opex_energy_initial_compression"]
            + opex_purification
        ) / (self.CO2_m_kg_per_s * self.operating_hours_per_a * 3.6)
        return levelized_costs

    def _levelized_cost(
        self, best_steel_grade_config, pinlet_mpa, poutlet_mpa, n_pump
    ):
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        print(f"_levelized_cost calculation ------------------------------")
        id_nps_m = best_steel_grade_config["id_nps_m"].iloc[0]
        v_m_per_s = self._calculate_velocity(id_nps_m)
        current_result = {}
        current_result["lc"] = 1e10

        if (v_m_per_s >= terrain_data["vRange_min"]) and (
            v_m_per_s <= terrain_data["vRange_max"]
        ):
            #Calculate Number of pumps
            delta_p_act_pa_m = self._calculate_actual_pressure_drop(id_nps_m)
            if self.phase == "liquid":                                                                                          #liquid recompression
                l_pump_m = self._calculate_max_distance_pumps(pinlet_mpa, poutlet_mpa, delta_p_act_pa_m, n_pump)
            else:                                                                                                               #gaseous recompression
                l_pump_m = self._calculate_max_distance_compressors(pinlet_mpa, poutlet_mpa, id_nps_m, n_pump)
            n_pumps = self._calculate_number_pumps(l_pump_m, self.terrain, n_pump)

            # RECOMPRESSION COST AND ENERGY
            if n_pumps == 0:
                # No recompression stations
                # e_comp_MJ_per_kg_all_but_last = 0
                # e_comp_MJ_per_kg_last = 0
                # e_comp_kJ_per_kg_total = 0
                capex_recompression_eur = 0
                opex_energy_recompression_eur_per_y = 0
                w_recompression_mw_total = 0
                e_comp_MJ_per_kg_total = 0
                p_outlet_last_pump_pa = 0
            else:

                p_outlet_last_pump_pa = self._calculate_pressure_last_pump(
                    l_pump_m, n_pumps, delta_p_act_pa_m
                )
                if self.phase == "gas":
                    p_ave_mpa = self._calculate_average_pressure(poutlet_mpa * 1e6, p_outlet_last_pump_pa) / 1e6
                    p_outlet_last_pump_pa = self._calculate_compressor_outlet_gas(
                        l_pump_m, p_ave_mpa, poutlet_mpa,  id_nps_m
                    )
                e_comp_MJ_per_kg_all_but_last = self._calculate_recompression_energy(
                    pinlet_mpa, poutlet_mpa
                )
                print(f"e_comp_MJ_per_kg_all_but_last: {e_comp_MJ_per_kg_all_but_last}")
                e_comp_MJ_per_kg_last = self._calculate_recompression_energy(
                    p_outlet_last_pump_pa * 1e-6, poutlet_mpa
                )
                print(f"e_comp_MJ_per_kg_last: {e_comp_MJ_per_kg_last}")
                e_comp_MJ_per_kg_total = (
                    e_comp_MJ_per_kg_last + e_comp_MJ_per_kg_all_but_last * n_pumps
                )
                w_recompressor_all_but_last_MW = (
                    e_comp_MJ_per_kg_all_but_last * self.total_m_kg_per_s
                )
                w_recompressor_last_MW = e_comp_MJ_per_kg_last * self.total_m_kg_per_s
                w_recompression_mw_total = e_comp_MJ_per_kg_total * self.total_m_kg_per_s

                capex_recompression_eur = self._calculate_compressor_cost(
                    w_recompressor_all_but_last_MW, self.phase
                ) * (n_pumps - 1) + self._calculate_compressor_cost(
                    w_recompressor_last_MW, self.phase
                )
                opex_energy_recompression_eur_per_y = (
                    self._calculate_recompression_energy_cost(
                        w_recompressor_all_but_last_MW
                    )
                    * (n_pumps - 1)
                    + self._calculate_recompression_energy_cost(w_recompressor_last_MW)
                )

            if self.terrain == "Onshore" and n_pumps != 0:                                                              #liquid/gas onshore with !=0 pumps -> recompression stations
                p_2 = pinlet_mpa
            else:                                                                                                     #liquid/gas with terrain offshore/onshore and 0 pumps -> no recompression station
                p_outlet_last_pump_pa = self._calculate_pressure_last_pump(
                    l_pump_m, n_pumps, delta_p_act_pa_m
                )
                if self.phase == "gas":
                    p_ave_mpa = self._calculate_average_pressure(poutlet_mpa * 1e6, p_outlet_last_pump_pa) / 1e6
                    p_outlet_last_pump_pa = self._calculate_compressor_outlet_gas(
                        l_pump_m, p_ave_mpa, poutlet_mpa, id_nps_m
                    )
                p_2 = p_outlet_last_pump_pa / 1000000

            p_max_mpa = terrain_data["PinletMAX_MPa"]
            if p_2 > p_max_mpa:
                print(f"Skipping: p_2={p_2} > {p_max_mpa}")
                return None


            # INITIAL COMPRESSION COST AND ENERGY
            (e_init_total_MJkg, e_init_gas_MJkg, e_init_pump_MJkg, dp_pump) = \
                self._calculate_initial_compression_energy(p_2, self.p_initial_mpa)
            w_init_gas_MW = e_init_gas_MJkg * self.total_m_kg_per_s
            w_init_pump_MW = e_init_pump_MJkg * self.total_m_kg_per_s
            w_initial_compression_mw = w_init_gas_MW + w_init_pump_MW

            capex_initial_gas_eur = self._calculate_compressor_cost(w_init_gas_MW, phase="gas")
            capex_initial_pump_eur = self._calculate_compressor_cost(w_init_pump_MW, phase="liquid") if dp_pump > 0 else 0.0
            capex_initial_compression_eur = capex_initial_gas_eur + capex_initial_pump_eur

            # Energy OPEX
            opex_energy_initial_compression_eur_per_y = self._calculate_recompression_energy_cost(
                w_initial_compression_mw
            )

            # Keep using your existing opex_fix multiplier later on:
            # current_result["opex_fix_compression"] = (capex_recompression_eur + capex_initial_compression_eur) * muOMpumpcomp
            # and the energy KPIs:
            e_initial_compression_Mj_per_kg = e_init_total_MJkg

            l_last_pump_km = self.length_km-l_pump_m*n_pumps/1000

            # CAPEX
            current_result["capex_pipe"] = best_steel_grade_config["capex_total"].iloc[
                0
            ]
            current_result["capex_recompression"] = capex_recompression_eur
            current_result["capex_initial_compression"] = capex_initial_compression_eur
            current_result["capex_total"] = (
                current_result["capex_pipe"]
                + current_result["capex_recompression"]
                + current_result["capex_initial_compression"]
            )

            # OPEX
            current_result["opex_pipe"] = best_steel_grade_config["opex_fix"].iloc[0]
            current_result["opex_energy_recompression"] = (
                opex_energy_recompression_eur_per_y
            )
            current_result["opex_energy_initial_compression"] = (
                opex_energy_initial_compression_eur_per_y
            )
            current_result["opex_fix_compression"] = (
                capex_recompression_eur + capex_initial_compression_eur
            ) * self.universal_data["muOMpumpcomp"]
            current_result["opex_fix"] = current_result["opex_fix_compression"]
            current_result["opex_var_energy"] = (
                current_result["opex_energy_recompression"]
                + current_result["opex_energy_initial_compression"]
            )

            # Energy consumption
            current_result["energy_compression_specific_Mj_per_kg"] = (
                e_comp_MJ_per_kg_total + e_initial_compression_Mj_per_kg
            )
            current_result["energy_compression_specific_MWh_per_kg"] = (
                current_result["energy_compression_specific_Mj_per_kg"] / 3.6 / 1000
            )
            current_result["energy_compression_specific_MWh_per_t"] = (
                current_result["energy_compression_specific_MWh_per_kg"] * 1000
            )

            terrain_data = self.terrain_specific_data[self.phase][self.terrain]
            density = terrain_data["rho_kg_per_m3"]
            viscosity = terrain_data["mu_Pas"]

            # Design
            current_result["steel_grade"] = best_steel_grade_config.index[0]
            current_result["n_pumps"] = n_pumps
            current_result["id_nps_m"] = id_nps_m
            current_result["l_pump_km"] = l_pump_m / 1000
            current_result["poutlet_mpa"] = poutlet_mpa
            current_result["pinlet_mpa"] = pinlet_mpa
            current_result["p_initial_mpa"] = self.p_initial_mpa
            current_result["p_2"] = p_2
            current_result["l_last_pump_km"] = l_last_pump_km
            current_result["p_outlet_last_pump_mpa"] = p_outlet_last_pump_pa / 1000000
            current_result["density"] = density
            current_result["viscosity"] = viscosity
            current_result["delta_p_act_pa_m"] = delta_p_act_pa_m
            current_result["design_p_inlet"] = pinlet_mpa
            current_result["t_m"] = best_steel_grade_config["t_m"].iloc[0]

            current_result["lc"] = self._calculate_levelized_cost(current_result)

        else:
            print(f"v not in boundaries")
        return current_result


class CO2Chain_Oeuvray(CO2Transport_Oeuvray):
    def _preprocess_data(self):
        super()._preprocess_data()
        # max number of pumps
        for phase in ["gas", "liquid"]:
            self.terrain_specific_data[phase]["Offshore"]["Npump_max"] = 0
            self.terrain_specific_data[phase]["Onshore"]["Npump_max"] = (self.length_km / 40)

    def calculate_cost(self, options):
        """
        Calculates the transport cost of CO2, including pipelines, compression and recompression

        :param str currency: currency of cost
        :param int year: year to use
        :param float discount_rate: discount rate
        :param str timeframe: determines which steel grades are available, can be 'short-term', 'mid-term', or 'long-term'
        :param float length_km: distance to cover in km
        :param float CO2_m_kg_per_s: mass flow rate of CO2 in kg/s
        :param float total_m_kg_per_s: total mass flow rate of the fluid in kg/s
        :param str terrain: 'Offshore' or 'Onshore'
        :param float electricity_price_eur_per_mw: used to minimize levelized cost (EUR/MWh)
        :param int operating_hours_per_a: number of operating hours per year
        :param float p_inlet_bar: inlet pressure in bar (beginning of pipeline)
        :param float p_outlet_bar: outlet pressure in bar (end of pipeline)
        :return: dictonary of cost and energy comsumption indicators of lowest levelized cost configuration
        :rtype: dict
        """
        self.length_km = options["length_km"]
        self.timeframe = options["timeframe"]
        self.CO2_m_kg_per_s = options["massflow_CO2_kg_per_s"]
        self.terrain = options["terrain"]
        self.electricity_price_eur_per_mw = options["electricity_price_eur_per_mw"]
        self.operating_hours_per_a = options["operating_hours_per_a"]
        self.p_initial_mpa = options["p_initial_bar"] / 10
        self.discount_rate = options["discount_rate"]
        self.total_m_kg_per_s = self._calculate_total_massflow(self.CO2_m_kg_per_s, self.flue_gas)

        default_result = 1e3
        self.current_best_results["lc"] = default_result
        if options["phase"] == "gas":
            self.phase = "gas"
        elif options["phase"] == "liquid":
            self.phase = "liquid"
        else:
            raise ValueError("Given fluid state is not supported. Please enter a fluid state of either 'gas' or 'liquid'")

        self._preprocess_data()

        if self.phase == "gas":
            # Starting values
            pinlet_mpa = 1.6
            poutlet_mpa = 1.5
            self.p_initial_mpa = 1.5
            self.p_outlet_system_mpa = 1.5
            id_calc_m = 0.5
            delta_p_inlet = 0.1
            optimal_configuration = self._calculate_pipeline_configuration(
                pinlet_mpa, poutlet_mpa, id_calc_m, delta_p_inlet
            )
        else:
            # Starting values
            poutlet_mpa = self.constants["Pcrit"] + 0.5
            if poutlet_mpa <= 8.5:
                pinlet_mpa = 9
            else:
                pinlet_mpa = poutlet_mpa + 1
            self.p_initial_mpa = 12
            self.p_outlet_system_mpa = 12
            if self.p_outlet_system_mpa < self.constants["Pcrit"] + 0.5:
                self.p_outlet_system_mpa = self.constants["Pcrit"] + 0.5
            id_calc_m = 0.5
            delta_p_inlet = 1
            optimal_configuration = self._calculate_pipeline_configuration(
                pinlet_mpa, poutlet_mpa, id_calc_m, delta_p_inlet
            )

        if optimal_configuration["lc"] == default_result:
            return None

        # Financial indicators
        self.optimal_configuration = optimal_configuration
        self.lifetime = min(self.universal_data)
        self.unit_capex = None
        self.opex_fix = None
        self.opex_var = None

        cost_pipeline = {}
        cost_pipeline["unit_capex"] = self.optimal_configuration["capex_pipe"]
        cost_pipeline["opex_var"] = 0
        cost_pipeline["opex_fix_abs"] = self.optimal_configuration[
            "opex_pipe"
        ]
        cost_pipeline["opex_fix_fraction"] = self.optimal_configuration["opex_pipe"] / (
            cost_pipeline["unit_capex"]
        )
        cost_pipeline["lifetime"] = self.universal_data["z_pipe"]

        cost_compression = {}
        cost_compression["unit_capex"] = (
            self.optimal_configuration["capex_total"]
            - self.optimal_configuration["capex_pipe"]
        )
        cost_compression["opex_var"] = 0
        cost_compression["opex_fix_abs"] = self.optimal_configuration[
            "opex_fix_compression"
        ]
        if self.optimal_configuration["opex_fix_compression"] and cost_compression["unit_capex"] != 0:
            cost_compression["opex_fix_fraction"] = (
                self.optimal_configuration["opex_fix_compression"]
                / cost_compression["unit_capex"]
            )
        else:
            cost_compression["opex_fix_fraction"] = 0
        cost_compression["lifetime"] = self.universal_data["z_pumpcomp"]

        energy_requirements = {}
        energy_requirements["specific_compression_energy"] = self.optimal_configuration[
            "energy_compression_specific_MWh_per_t"
        ]

        results = {}
        results["cost_pipeline"] = cost_pipeline
        results["cost_compression"] = cost_compression
        results["energy_requirements"] = energy_requirements
        results["configuration"] = self.optimal_configuration
        results["levelized_cost"] = self.optimal_configuration["lc"]
        results["steel_grade"] = self.optimal_configuration.get("steel_grade", None)
        results["inner_diameter_m"] = self.optimal_configuration.get("id_nps_m", None)
        results["p_2_MPa"] = self.optimal_configuration.get("p_2", None)
        results["p_outlet_last_pump_mpa"] = self.optimal_configuration.get("p_outlet_last_pump_mpa", None)
        results["poutlet_mpa"] = self.optimal_configuration.get("poutlet_mpa", None)
        results["n_pumps"] = self.optimal_configuration.get("n_pumps", None)
        results["l_pump_km"] = self.optimal_configuration.get("l_pump_km", None)
        results["l_last_pump_km"] = self.optimal_configuration.get("l_last_pump_km", None)
        results["density"] = self.optimal_configuration.get("density", None)
        results["viscosity"] = self.optimal_configuration.get("viscosity", None)
        results["delta_p_act_pa_m"] = self.optimal_configuration.get("delta_p_act_pa_m", None)
        results["capex_pipe"] = self.optimal_configuration["capex_pipe"]
        results["capex_recompression"] = self.optimal_configuration["capex_recompression"]
        results["capex_initial_compression"] = self.optimal_configuration["capex_initial_compression"]
        results["opex_pipe"] = self.optimal_configuration["opex_pipe"]
        results["opex_fix_compression"] = self.optimal_configuration["opex_fix_compression"]
        results["opex_energy_recompression"] = self.optimal_configuration["opex_energy_recompression"]
        results["opex_energy_initial_compression"] = self.optimal_configuration["opex_energy_initial_compression"]
        results["design_p_inlet"] = self.optimal_configuration.get("design_p_inlet")
        results["t_m"] = self.optimal_configuration.get("t_m", None)
        print(f"[DEBUG] Returning results['l_last_pump_km']: {results['l_last_pump_km']}")

        return results


class CO2Compression_Oeuvray(CO2Transport_Oeuvray):
    """
    Calculates the compressor costs and specific compression energy isolated from the CO2 transport chain

    - unit_capex is in [selected currency] / kg(CO2) / s
    - opex_fix is in % of up-front capex
    - opex_var is 0
    - specific_compression_energy_mwh_per_t is in MWh/t(CO2)
    """

    def __init__(self, fluid_properties_file="CO2IsothermalProperties.xlsx"):
        super().__init__()
        input_path = Path(__file__).parent.parent.parent.parent / Path(
            "./data/networks/co2_transport_oeuvray/"
        )
        fluid_properties_input_path = input_path / fluid_properties_file
        universal_data_input_path = input_path / "OtherData.xlsx"

        self.poutlet_mpa = None
        self.p_inlet_mpa = None

    def calculate_cost(self, options):
        """
        Calculates cost and compression energy (only for prior conditioning, pumping stations not included)


        :param str currency: currency of cost
        :param int year: year to use
        :param float discount_rate: discount rate
        :param float m_kg_per_s: mass flow rate of CO2 in kg/s
        :param float p_inlet_bar: inlet pressure in bar (beginning of pipeline)
        :param float p_outlet_bar: outlet pressure in bar (end of pipeline)
        :return: dictonary of cost and energy comsumption indicators
        :rtype: dict
        """
        self.m_kg_per_s = options["m_kg_per_s"]
        self.p_inlet_mpa = options["p_inlet_bar"] / 10
        self.poutlet_mpa = options["p_outlet_bar"] / 10
        self.discount_rate = options["discount_rate"]
        self.terrain = options["terrain"]
        #print(f"Mass flow: {self.m_kg_per_s}")                                                     #CCC

        self._preprocess_data()

        # Determine phase
        if self.poutlet_mpa < 3:                                                             #3e6 mpa?? Mistake in prev version
            self.phase = "gas"                                                                       #CCC
        elif self.poutlet_mpa >= 3:
            self.phase = "liquid"

        print(f"self.poutlet_mpa: {self.poutlet_mpa}")                                    #CCC
        print(f"self.phase: {self.phase}")

        # COMPRESSION COST AND ENERGY
        e_initial_compression_Mj_per_kg = self._calculate_initial_compression_energy(
            self.poutlet_mpa, self.p_inlet_mpa
        )
        w_compression_mw = e_initial_compression_Mj_per_kg * self.m_kg_per_s
        w_compression_mwh_per_t = e_initial_compression_Mj_per_kg / 3.6

        capex_compression_eur = self._calculate_compressor_cost(
            w_compression_mw, phase="gas"                                                               #gas??
        )
        #print(f"Capex: {capex_compression_eur}")                                                  #CCC

        opex_fix = capex_compression_eur * self.universal_data["muOMpumpcomp"]

        cr_pump_compressions = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.universal_data["z_pumpcomp"]
            / ((1 + self.discount_rate) ** self.universal_data["z_pumpcomp"] - 1)
        )

        self.lifetime = min(self.universal_data)                                                            #XXX
        self.unit_capex = capex_compression_eur
        self.opex_fix = opex_fix / capex_compression_eur                                                    #XXX
        self.opex_var = 0

        return {
            "unit_capex": self.unit_capex,
            "opex_fix": self.opex_fix,
            "opex_var": self.opex_var,
            "specific_compression_energy_mwh_per_t": w_compression_mwh_per_t,
        }
