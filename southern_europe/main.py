import adopt_net0 as adopt
import json
from pathlib import Path
import os
import pandas as pd

# Scenario parameterisation
ref_year_network = 2024 # possible choices [2024, 2030,2040, 2050]
inflation_rate = 0.011 # annual inflation rate provided by European Commission
new_technology_size_max = 5395910562 # total emission of the selected nodes
carbon_tax = 0

# Create folder for results
results_data_path = Path("./results")
results_data_path.mkdir(parents=True, exist_ok=True)
# Create input data path and optimisation templates
input_data_path = Path("northern_italy_case")
input_data_path.mkdir(parents=True, exist_ok=True)
adopt.create_optimization_templates(input_data_path)

# Import data
path_data_case_study = Path("./northern_italy_data")
input_data_path.mkdir(parents=True, exist_ok=True)
network_location = pd.read_excel(path_data_case_study/"geographical_feature/node_matrics.xlsx", index_col=0, sheet_name='nodes')# nodes
network_distance =pd.read_excel(path_data_case_study/"geographical_feature/node_matrics.xlsx", index_col=0, sheet_name='distances')# path distance
network_emission_flux = pd.read_excel(path_data_case_study/"geographical_feature/node_matrics.xlsx", index_col=0, sheet_name='nodes') # annual emission fluxes
network_storage_capacity = pd.read_excel(path_data_case_study/"geographical_feature/node_matrics.xlsx", index_col=0, sheet_name='storage_capacity') # storage capacity
network_pipeline = pd.read_excel(path_data_case_study/"geographical_feature/node_matrics.xlsx", index_col=0, sheet_name='pipeline_transport')# pipeline connection
network_truck = pd.read_excel(path_data_case_study/"geographical_feature/node_matrics.xlsx", index_col=0, sheet_name='truck_transport')# truck connection
network_train = pd.read_excel(path_data_case_study/"geographical_feature/node_matrics.xlsx", index_col=0, sheet_name='train_transport')# train connection
morphological_feature_grid = pd.read_csv(path_data_case_study/"geographical_feature/morphological_feature_grids_italy.csv")# morphological feature by grid
soil_type_grid = pd.read_csv(path_data_case_study/"geographical_feature/soil_type_grids_italy.csv")# soil type by grid
anthropisation_grid = pd.read_csv(path_data_case_study/"geographical_feature/anthropisation_grids_italy.csv")# anthropisation by grid
electricity_price = pd.read_csv(path_data_case_study/"electricity_metrics/electricity_prices_hourly_2024.csv")# electricity price
co2_intensity_electricity = pd.read_excel(path_data_case_study/"")
node_names = network_location['node_name'].tolist() # all nodes

# Update network cost ?
# co2 pipeline cost updated

# Update CCS technology cost ?
# capture technology
# storage technology

# Update topology json
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

# Define node locations
adopt.create_input_data_folder_template(input_data_path)
node_location = pd.read_csv(input_data_path / "NodeLocations.csv", sep=';', index_col=0, header=0)

for node in node_names:
    node_location.at[node, 'lon'] = network_location.at[node, "longitude"]
    node_location.at[node, 'lat'] = network_location.at[node, "latitude"]
    node_location.at[node, 'alt'] = network_location.at[node, "altitude"] ? add

node_location = node_location.reset_index()
node_location.to_csv(input_data_path / "NodeLocations.csv", sep=';', index=False)

# Add technologies for emitter nodes ?
adopt.show_available_technologies()
# cement+waste: post-combustion

# Add technologies for storage site nodes
#permanentstoragesimple ?

# (Add) technologies for transport nodes
#no ?

# Copy over technology files
adopt.copy_technology_data(input_data_path, #technologies folder ?)

# Distance ? matrix
istance = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "distance.csv", sep=";", index_col=0)
for node_x in node_names:
    for node_y in node_names:
        if node_x != node_y:
            distance.loc[node_x, node_y] = network_distances.at[node_x, node_y]
            distance.loc[node_y, node_x] = network_distances.at[node_x, node_y]
# distance.to_csv(input_data_path / "period1" / "network_topology" / "existing" / "electricityOnshore" / "distance.csv", sep=";")
print("Distance:", distance)

# Delete the template
os.remove(input_data_path / "period1" / "network_topology" / "existing" / "distance.csv")

# Size ? new instead of existing, no size for existing, size_max=total_emission
size = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "size.csv", sep=";", index_col=0)
for node_x in node_names:
    for node_y in node_names:
        if node_x != node_y:
            size.loc[node_x, node_y] = network_emission_flux.at[node_x, node_y]
            size.loc[node_y, node_x] = network_emission_flux.at[node_y, node_x]
# size.to_csv(input_data_path / "period1" / "network_topology" / "existing" / "electricityOnshore" / "size.csv", sep=";")
print("Size:", size)

# Delete the template
os.remove(input_data_path / "period1" / "network_topology" / "existing" / "size.csv")

