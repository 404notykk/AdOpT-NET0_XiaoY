import adopt_net0 as adopt
import json
from pathlib import Path
import os
import pandas as pd
from data_process.utilities.defined_functions import (
    calculate_annual_emission_values,
    assign_mea_technology,
    assign_ccs_technologies,
    update_network_distance_matrix,
    update_network_connection_matrix,
    update_network_size_max_arcs,
    update_network_size_min_arcs,
    calculate_production_profiles,
    create_demand_profiles
)

#----- Scenario parameterization -----#
ref_year = 2024
simulation_year = 2030 # possible choices [2024, 2030, 2040, 2050]
discount_rate = 0.1 # default
size_max = 5395910562 # total emission of the selected nodes ? summing up annual_emission instead

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
path_files_node_flux = path_data_case_study / "geographical_feature"
path_files_electricity = path_data_case_study / "electricity_metrics"

network_location = pd.read_excel(path_files_node_flux/"node_metrics.xlsx", index_col=0, sheet_name='nodes') # nodes
network_distance =pd.read_excel(path_files_node_flux/"node_metrics.xlsx", index_col=0, sheet_name='distances') # path distance
network_emission_flux = pd.read_excel(path_files_node_flux/"node_metrics.xlsx", index_col=0, sheet_name='nodes') # annual emission fluxes
network_pipeline = pd.read_excel(path_files_node_flux/"node_metrics.xlsx", index_col=0, sheet_name='pipeline_transport') # pipeline connection
network_truck = pd.read_excel(path_files_node_flux/"node_metrics.xlsx", index_col=0, sheet_name='truck_transport') # truck connection
network_train = pd.read_excel(path_files_node_flux/"node_metrics.xlsx", index_col=0, sheet_name='train_transport') # train connection
network_storage_capacity = pd.read_excel(path_files_node_flux/"storage_capacity_by_scenario.xlsx", index_col=0, sheet_name='storage_capacity') # storage capacity ? not defined

electricity_price = pd.read_csv(path_files_electricity/"electricity_prices_hourly_2024.csv") # electricity price
co2_intensity_electricity = pd.read_excel(path_files_electricity/"co2_intensity_by_scenario.xlsx", index_col=0, sheet_name='co2_intensity_electricity') # co2 intensity of electricity ? not defined

node_names = network_location['node_name'].unique().tolist() # all nodes

#----- Calculate annual emission values -----#
# Calculate the actual annual emission values using the specified formula logic
network_emission_flux = calculate_annual_emission_values(network_emission_flux)

#----- Calculate production profiles from emissions -----#
production_profiles = calculate_production_profiles(network_emission_flux, path_data_case_study)
print(production_profiles)

#----- Update network cost ? -----#
# co2 pipeline cost updated

#----- Update topology json -----#
with open(input_data_path / "Topology.json", "r") as json_file:
    topology = json.load(json_file)
topology["nodes"] = node_names # nodes
topology["carriers"] = ["electricity", "CO2captured", "heat"]
topology["investment_periods"] = ["period1"] # investment periods
with open(input_data_path / "Topology.json", "w") as json_file:
    json.dump(topology, json_file, indent=4)

# Update configmodel json
with open(input_data_path / "ConfigModel.json", "r") as json_file:
    configuration = json.load(json_file)
configuration["optimization"]["objective"]["value"] = "costs" # set optimisation objective
configuration["solveroptions"]["mipgap"]["value"] = 0.02 # set MILP gap
with open(input_data_path / "ConfigModel.json", "w") as json_file:
    json.dump(configuration, json_file, indent=4)

#----- Define node locations -----#
adopt.create_input_data_folder_template(input_data_path)
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

#----- Create demand profiles ? -----#
create_demand_profiles(production_profiles, input_data_path, carriers=['electricity', 'heat'])

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

# Distance (all networks)
update_network_distance_matrix(input_data_path, network_distance, new_network_types)

# Dictionary mapping network types to their connection data
connection_data_mapping = {
    'CO2_Pipeline': network_pipeline,
    'CO2Truck': network_truck,
    'CO2Railway': network_train
}

# Connection (all networks)
update_network_connection_matrix(input_data_path, connection_data_mapping)

# Max size arc (all networks) - using predefined size_max value
update_network_size_max_arcs(input_data_path, connection_data_mapping, size_max)

# Min size arc (all networks) - now using calculated annual_emission values
update_network_size_min_arcs(input_data_path, connection_data_mapping, network_emission_flux)

# Delete the template
os.remove(input_data_path / "period1" / "network_topology" / "new" / "distance.csv")
os.remove(input_data_path / "period1" / "network_topology" / "new" / "connection.csv")
os.remove(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv")

# Copy network data and change costs ??? folder
adopt.copy_network_data(input_data_path)

#----- Change network economic parameters ? -----#

# #----- Define climate data (default?) -----# <- see error in clipboard.py
# adopt.load_climate_data_from_api(input_data_path)

# #----- Build and solve optimization problem ? need to be activated -----#
# m = adopt.ModelHub()
# m.read_data(input_data_path)
# m.quick_solve()