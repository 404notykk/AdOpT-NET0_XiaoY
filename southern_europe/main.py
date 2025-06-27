import adopt_net0 as adopt
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
from data_process.utilities.defined_functions import (
    calculate_annual_emission_values,
    assign_carriers_to_nodes,
    assign_mea_technology,
    assign_ccs_technologies,
    update_network_distance_matrix,
    update_network_connection_matrix,
    update_network_size_max_arcs,
    update_network_size_min_arcs,
    load_climate_data_from_api_robust,
    update_carrier_data,
    process_gamma_sheets_to_csv
)


#----- Scenario parameterization -----#
ref_year = 2024
simulation_year = 2030 # possible choices [2024, 2030, 2040, 2050]
discount_rate = 0.1 # default
co2_intensity_electricity = 0.288 # default (kg CO2/kWh)
heat_convert_factor = 2.6 # default
electricity_import_limit = 100 # default
heat_import_limit = 200 # default

#----- Create folder for results -----#
results_data_path = Path("./results")
results_data_path.mkdir(parents=True, exist_ok=True)
# Create input data path and optimisation templates
input_data_path = Path("northern_italy_case")
input_data_path.mkdir(parents=True, exist_ok=True)
adopt.create_optimization_templates(input_data_path)

#----- Import data-----#
path_data_case_study = Path("./northern_italy_data")
input_data_path.mkdir(parents=True, exist_ok=True)

path_files_technologies = path_data_case_study / "technologies"
path_files_networks = path_data_case_study / "networks"
path_files_node_flux = path_data_case_study / "geographical_feature"
path_files_electricity = path_data_case_study / "electricity_metrics"
path_files_network_capex = path_data_case_study / "network_capex_metrics"

network_location = pd.read_excel(path_files_node_flux/"node_metrics.xlsx", index_col=0, sheet_name='nodes') # nodes
network_emission_flux = pd.read_excel(path_files_node_flux/"node_metrics.xlsx", index_col=0, sheet_name='nodes') # annual emission fluxes
network_pipeline = pd.read_excel(path_files_node_flux/"node_metrics.xlsx", index_col=0, sheet_name='pipeline') # pipeline connection and distance
network_truck = pd.read_excel(path_files_node_flux/"node_metrics.xlsx", index_col=0, sheet_name='truck') # truck connection and distance
network_railway = pd.read_excel(path_files_node_flux/"node_metrics.xlsx", index_col=0, sheet_name='railway') # train connection and distance
# network_storage_capacity = pd.read_excel(path_files_node_flux/"storage_capacity_by_scenario.xlsx", index_col=0, sheet_name='storage_capacity') # storage capacity ? not defined

electricity_price = pd.read_csv(path_files_electricity/"electricity_prices_hourly_2024.csv") # electricity price

node_names = network_location['node_name'].unique().tolist() # all nodes

#----- Calculate annual emission values -----#
# Calculate the actual annual emission values using the specified formula logic
network_emission_flux = calculate_annual_emission_values(network_emission_flux)

#----- Update topology json with carriers assignment -----#
adopt.create_input_data_folder_template(input_data_path)

# Assign carriers to nodes based on their types
assign_carriers_to_nodes(input_data_path, network_location, network_emission_flux)

# Update configmodel json
with open(input_data_path / "ConfigModel.json", "r") as json_file:
    configuration = json.load(json_file)
configuration["optimization"]["objective"]["value"] = "costs" # set optimisation objective
configuration["solveroptions"]["mipgap"]["value"] = 0.02 # set MILP gap
with open(input_data_path / "ConfigModel.json", "w") as json_file:
    json.dump(configuration, json_file, indent=4)

#----- Define node locations -----#
node_location = pd.read_csv(input_data_path / "NodeLocations.csv", sep=';', index_col=0, header=0)

for node in node_names:
    node_row = network_location[network_location['node_name'] == node]
    if not node_row.empty:
        node_location.at[node, 'lon'] = node_row['longitude'].values[0]
        node_location.at[node, 'lat'] = node_row['latitude'].values[0]
        node_location.at[node, 'alt'] = node_row['altitude'].values[0]
    else:
        print(f"Warning: Node {node} not found in network_location dataframe")

node_location = node_location.reset_index()
node_location.to_csv(input_data_path / "NodeLocations.csv", sep=';', index=False)

#----- Add technologies for nodes -----#
# Assign MEA technology to network_emission_flux (now using calculated annual_emission values)
network_emission_flux = assign_mea_technology(network_emission_flux, path_data_case_study)

# Then assign CCS technologies, passing both DataFrames
assign_ccs_technologies(network_location, network_emission_flux, path_data_case_study, input_data_path)

# Copy over technology files
adopt.copy_technology_data(input_data_path, path_files_technologies)

#----- Add networks -----#
new_network_types = ["CO2_Pipeline", "CO2Truck", "CO2Railway"]

with open(input_data_path / "period1" / "Networks.json", "r") as json_file:
    networks = json.load(json_file)
networks["new"] = new_network_types

with open(input_data_path / "period1" / "Networks.json", "w") as json_file:
    json.dump(networks, json_file, indent=4)

# Since there are no existing networks, simply remove all template from the network folder
os.remove(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv")
os.remove(input_data_path / "period1" / "network_topology" / "existing" / "distance.csv")
os.remove(input_data_path / "period1" / "network_topology" / "existing" / "size.csv")

# Make folders for the new networks
for network_type in new_network_types:
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / network_type, exist_ok=True)

# Prepare network data dictionary for the updated functions
# Each matrix contains values where: 0 = no connection, >0 = connected with distance value
network_data_dict = {
    'pipeline': network_pipeline,
    'truck': network_truck,
    'railway': network_railway
}

# Distance matrices (use actual values from network data)
update_network_distance_matrix(input_data_path, network_data_dict, new_network_types)

# Connection matrices (convert >0 to 1, keep 0 as 0)
update_network_connection_matrix(input_data_path, network_data_dict)

# Max size arc (all networks) - using predefined size_max value
size_max = network_emission_flux['annual_emission'].sum() # total emission of the selected nodes
update_network_size_max_arcs(input_data_path, network_data_dict, size_max)

# Min size arc (all networks) - now using calculated annual_emission values
update_network_size_min_arcs(input_data_path, network_data_dict, network_emission_flux)

# Delete the templates
os.remove(input_data_path / "period1" / "network_topology" / "new" / "distance.csv")
os.remove(input_data_path / "period1" / "network_topology" / "new" / "connection.csv")
os.remove(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv")

# Copy network data and change costs
adopt.copy_network_data(input_data_path, path_files_networks)

#----- Process gamma sheets from capex_defined_per_arc.xlsx -----#
# Process gamma sheets and save as CSV files in CO2_Pipeline folder
gamma_pipeline_per_arc = process_gamma_sheets_to_csv(path_files_network_capex, input_data_path)

#----- Update carrier data with pricing, emission factors, and demands -----#
update_carrier_data(
    input_data_path,
    electricity_price,
    network_emission_flux,
    path_files_technologies,
    node_names,
    co2_intensity_electricity,
    heat_convert_factor,
    electricity_import_limit,
    heat_import_limit
)

#----- Define climate data -----#
load_climate_data_from_api_robust(input_data_path)

#----- Build and solve optimization problem ? need to be activated -----#
m = adopt.ModelHub()
m.read_data(input_data_path)
m.quick_solve()