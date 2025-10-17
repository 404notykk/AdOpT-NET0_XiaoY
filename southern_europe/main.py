import adopt_net0 as adopt
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
import json
from data_process.utilities.defined_functions import (
    calculate_annual_emission_values,
    calculate_emitter_capacities,
    assign_carriers_to_nodes,
    update_network_distance_matrix_debug,
    update_network_connection_matrix,
    update_network_size_max_arcs,
    load_climate_data_from_api_robust,
    update_carrier_data,
    process_gamma_sheets_to_csv,
    copy_technology_data_custom,
    convert_network_data_indices_to_names,
    assign_ccs_technologies_debug,
    apply_carbon_pricing_to_all_nodes
)


#----- Scenario parameterization -----#
ref_year = 2024
discount_rate = 0.1 # default
co2_intensity_electricity = 0.288 # default (kg CO2/kWh)
heat_convert_factor = 2.6 # default
electricity_import_limit = 100 # default
heat_import_limit = 200 # default
max_transport_capacity = 3000
carbon_tax = 100  # euro per tonne CO2
enable_carbon_pricing = True
cpu_type = "CPU1"

#----- Create folder for results -----#
results_data_path = "./resultsImpurities"
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



electricity_price = pd.read_csv(path_files_electricity/"electricity_prices_hourly_2024.csv") # electricity price

node_names = network_location['node_name'].unique().tolist() # all nodes

#----- Calculate annual emission values -----#
# Calculate the actual annual emission values using the specified formula logic
network_emission_flux = calculate_annual_emission_values(network_emission_flux)

#----- Calculate emitter capacities -----#
# Calculate initial capacities for emitter technologies based on annual emissions and emission factors
# Using tonnes/hour units (appropriate for emitters that produce physical products)
print("Calculating emitter capacities based on annual emissions and emission factors...")
network_emission_flux = calculate_emitter_capacities(network_emission_flux, path_data_case_study, path_files_node_flux, capacity_unit="tonnes_per_hour")

#----- Update topology json with carriers assignment -----#
adopt.create_input_data_folder_template(input_data_path)

# Assign carriers to nodes based on their types
assign_carriers_to_nodes(input_data_path, network_location, network_emission_flux)

# Update configmodel json
with open(input_data_path / "ConfigModel.json", "r") as json_file:
    configuration = json.load(json_file)
configuration["optimization"]["objective"]["value"] = "emissions_minC" # set optimization objective (Options: emissions_minC; costs)
configuration["solveroptions"]["mipgap"]["value"] = 0.02 # set MILP gap
configuration['reporting']['save_summary_path']['value'] = results_data_path
configuration['reporting']['save_path']['value'] = results_data_path
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
# Then assign CCS technologies, passing both DataFrames
# Note: This now uses the calculated capacities from calculate_emitter_capacities()
assign_ccs_technologies_debug(network_location, network_emission_flux, path_data_case_study, input_data_path, cpu_type)







# Copy over technology files using our custom function
copy_technology_data_custom(input_data_path, path_files_technologies, network_emission_flux)


#----- Add networks -----#
new_network_types = ["CO2_Pipeline"]

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
}

print("\n🔍 Converting network data indices to match topology...")
network_data_dict = convert_network_data_indices_to_names(network_data_dict, network_location)


# Distance matrices (use actual values from network data)
update_network_distance_matrix_debug(input_data_path, network_data_dict, new_network_types)

# Connection matrices (convert >0 to 1, keep 0 as 0)
update_network_connection_matrix(input_data_path, network_data_dict)

# Max size arc (all networks) - using predefined size_max value
print(f"🔍 Network sizing: Using predefined transport capacity = {max_transport_capacity} tonnes/hour")
update_network_size_max_arcs(input_data_path, network_data_dict, max_transport_capacity)



# Delete the templates
os.remove(input_data_path / "period1" / "network_topology" / "new" / "distance.csv")
os.remove(input_data_path / "period1" / "network_topology" / "new" / "connection.csv")
os.remove(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv")

# Copy network data and change costs
adopt.copy_network_data(input_data_path, path_files_networks)

#----- Process gamma sheets from capex_defined_per_arc.xlsx -----#
# Process gamma sheets and save as CSV files in network folder
gamma_pipeline_per_arc = process_gamma_sheets_to_csv(
    path_files_network_capex,
    input_data_path,
    network_location,
    transport_mode="pipeline"
)



#----- Update carrier data with pricing, emission factors, and demands -----#
print("Updating carrier data with hourly demand profiles...")
update_carrier_data(
    input_data_path,
    electricity_price,
    network_emission_flux,
    path_files_technologies,
    node_names,
    co2_intensity_electricity,
    heat_convert_factor,
    path_files_node_flux,
    electricity_import_limit,
    heat_import_limit
)

# ----- Apply Carbon Pricing -----#
if enable_carbon_pricing and carbon_tax > 0:
    print(f"\n" + "=" * 60)
    print(f"💰 APPLYING CARBON PRICING: €{carbon_tax}/tonne CO2")
    print(f"=" * 60)

    # Apply carbon pricing to all nodes
    carbon_pricing_success = apply_carbon_pricing_to_all_nodes(
        input_data_path,
        carbon_tax,
        node_names
    )

else:
    print(f"\n💡 Carbon pricing disabled (carbon_tax={carbon_tax}, enabled={enable_carbon_pricing})")

#----- Define climate data -----#
load_climate_data_from_api_robust(input_data_path)

m = adopt.ModelHub()
m.read_data(input_data_path)
m.quick_solve()
