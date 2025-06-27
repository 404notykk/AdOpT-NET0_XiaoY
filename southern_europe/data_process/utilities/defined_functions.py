import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime


def calculate_annual_emission_values(network_emission_flux):
    """
    Calculate the actual annual emission values for each emitter based on Excel formula logic.

    This function recreates the Excel formula: =IF(ISNUMBER(computed_annual_flux), computed_annual_flux, annual_flux)
    If 'computed_annual_flux' has a valid number, use it; otherwise use 'annual_flux'.

    Parameters:
        - network_emission_flux: DataFrame containing emission data with 'annual_flux' and 'computed_annual_flux' columns

    Returns:
        - network_emission_flux: Updated DataFrame with calculated 'annual_emission' column
    """

    def apply_emission_formula(row):
        """
        Apply the Excel formula logic: =IF(ISNUMBER(computed_annual_flux),computed_annual_flux,annual_flux)
        If 'computed_annual_flux' has a valid number, use it; otherwise use 'annual_flux'
        """
        computed_flux = row.get('computed_annual_flux', None)
        annual_flux = row.get('annual_flux', 0)

        # Check if computed_flux is a valid number (not NaN, not None, not empty, not zero)
        if pd.notna(computed_flux) and computed_flux != 0:
            return computed_flux
        else:
            return annual_flux

    # Apply the logic to create the annual_emission column
    network_emission_flux['annual_emission'] = network_emission_flux.apply(apply_emission_formula, axis=1)

    return network_emission_flux


def assign_carriers_to_nodes(input_data_path, network_location, network_emission_flux):
    """
    Assign appropriate carriers to each node based on their type(s).

    Carrier assignment rules:
    - All nodes get: electricity, heat, CO2captured (except Transport nodes)
    - Transport nodes get: electricity, CO2captured only (no heat)
    - Cement nodes also get: cement
    - Waste nodes also get: waste
    - Other nodes also get: industrial_product
    - Nodes with multiple emitters get all relevant carriers from both emitter types

    Parameters:
        - input_data_path: Path to the input data directory
        - network_location: DataFrame containing node information with node_name and node_type
        - network_emission_flux: DataFrame containing emission data with node_name and node_type

    Returns:
        - None (updates Topology.json file)
    """

    # Get all unique nodes
    all_nodes = network_location['node_name'].unique().tolist()

    # Base carriers that most nodes get
    base_carriers = ["electricity", "heat", "CO2captured"]
    transport_carriers = ["electricity", "CO2captured"]  # Transport nodes don't get heat

    # Mapping from emitter node_type to specific carriers
    emitter_carriers = {
        'Cement': 'cement',
        'Waste': 'waste',
        'Other': 'industrial_product'
    }

    # Collect all unique carriers needed
    all_carriers = set(base_carriers)

    # Add emitter-specific carriers
    for node_name in all_nodes:
        # Check if this node has emitters in network_emission_flux
        node_emission_rows = network_emission_flux[network_emission_flux['node_name'] == node_name]

        for _, emission_row in node_emission_rows.iterrows():
            emitter_type = emission_row['node_type']
            if emitter_type in emitter_carriers:
                all_carriers.add(emitter_carriers[emitter_type])

    # Convert to sorted list for consistent output
    all_carriers = sorted(list(all_carriers))

    # Update Topology.json
    with open(input_data_path / "Topology.json", "r") as json_file:
        topology = json.load(json_file)

    topology["nodes"] = all_nodes
    topology["carriers"] = all_carriers
    topology["investment_periods"] = ["period1"]

    with open(input_data_path / "Topology.json", "w") as json_file:
        json.dump(topology, json_file, indent=4)

    # Log carrier assignment summary
    print(f"Carrier assignment completed:")
    print(f"  - Total nodes: {len(all_nodes)}")
    print(f"  - Total carriers: {all_carriers}")

    # Show detailed assignment for verification
    node_carrier_summary = {}
    for node_name in all_nodes:
        # Get node type(s) from network_location
        node_location_rows = network_location[network_location['node_name'] == node_name]
        node_emission_rows = network_emission_flux[network_emission_flux['node_name'] == node_name]

        # Determine carriers for this node
        node_carriers = set()

        # Check if any location row is Transport type
        is_transport = any(row['node_type'] == 'Transport' for _, row in node_location_rows.iterrows())

        if is_transport:
            node_carriers.update(transport_carriers)
        else:
            node_carriers.update(base_carriers)

        # Add emitter-specific carriers
        for _, emission_row in node_emission_rows.iterrows():
            emitter_type = emission_row['node_type']
            if emitter_type in emitter_carriers:
                node_carriers.add(emitter_carriers[emitter_type])

        node_carrier_summary[node_name] = sorted(list(node_carriers))

    # Show summary by node type
    transport_nodes = []
    emitter_nodes = []
    storage_nodes = []

    for node_name in all_nodes:
        node_types = set()
        # Get types from both DataFrames
        for _, row in network_location[network_location['node_name'] == node_name].iterrows():
            node_types.add(row['node_type'])
        for _, row in network_emission_flux[network_emission_flux['node_name'] == node_name].iterrows():
            node_types.add(row['node_type'])

        if 'Transport' in node_types:
            transport_nodes.append(node_name)
        elif any(t in ['Cement', 'Waste', 'Other'] for t in node_types):
            emitter_nodes.append(node_name)
        elif 'Storage' in node_types:
            storage_nodes.append(node_name)

    print(f"  - Transport nodes ({len(transport_nodes)}): {transport_nodes}")
    print(f"  - Emitter nodes ({len(emitter_nodes)}): {emitter_nodes}")
    print(f"  - Storage nodes ({len(storage_nodes)}): {storage_nodes}")

    return True


def assign_mea_technology(network_emission_flux, path_data_case_study):
    """
    Determines appropriate MEA (Monoethanolamine) carbon capture technology scale
    for emitter nodes based on their annual CO2 emissions.

    This function analyzes emission data for each node and determines the appropriate
    MEA technology scale (small, medium, large), adding it to a new column 'mea_technology'.

    Parameters:
        - network_emission_flux: DataFrame containing node information and emission data with 'annual_emission' column
        - path_data_case_study: Path to the case study data directory

    Returns:
        - network_emission_flux: Updated DataFrame with mea_technology column added
    """
    # Ensure annual_emission column exists
    if 'annual_emission' not in network_emission_flux.columns:
        raise ValueError("'annual_emission' column not found. Please run calculate_annual_emission_values() first.")

    # Define paths to different MEA technology scales
    mea_paths = {
        "large": path_data_case_study / "technologies/CCSTechnologies/MEA_large.json",
        "medium": path_data_case_study / "technologies/CCSTechnologies/MEA_medium.json",
        "small": path_data_case_study / "technologies/CCSTechnologies/MEA_small.json"
    }

    # Load MEA technology specifications from JSON files
    mea_data = {}
    for scale, path in mea_paths.items():
        with open(path, "r") as f:
            mea_data[scale] = json.load(f)

    # Add column for MEA technology if it doesn't exist
    network_emission_flux['mea_technology'] = None

    # Process each row in the network_emission_flux DataFrame
    for idx, row in network_emission_flux.iterrows():
        node_name = row['node_name']
        node_type = row['node_type']

        # Skip non-emitter nodes (Storage and Transport)
        if node_type in ["Storage", "Transport"]:
            continue

        # Get the node's calculated annual CO2 emission (kg/year)
        annual_emission = row["annual_emission"]

        # Determine CO2 concentration based on emitter type
        if node_type in ["Waste"]:
            co2_concentration = 0.07
        elif node_type in ["Cement"]:
            co2_concentration = 0.20
        else:
            co2_concentration = 0.15

        # Calculate CO2 ranges for each MEA scale based on technology specs
        # Convert MEA scale from t/h to kg/year for comparison
        conversion_factor = 1000 * 24 * 365  # t/h to kg/year

        mea_ranges = {}
        for scale, data in mea_data.items():
            min_co2 = co2_concentration * data["size_min"] * conversion_factor
            max_co2 = co2_concentration * data["size_max"] * conversion_factor
            mea_ranges[scale] = (min_co2, max_co2)

        # Find the MEA scale that matches the node's emission range
        suitable_mea = None
        for scale, (min_co2, max_co2) in mea_ranges.items():
            if min_co2 <= annual_emission <= max_co2:
                suitable_mea = scale
                break

        # If no exact match found, choose the closest scale
        if suitable_mea is None:
            distances = {}
            for scale, (min_co2, max_co2) in mea_ranges.items():
                if annual_emission < min_co2:
                    distances[scale] = min_co2 - annual_emission
                elif annual_emission > max_co2:
                    distances[scale] = annual_emission - max_co2

            suitable_mea = min(distances, key=distances.get)

        # Store the suitable MEA technology in the mea_technology column
        mea_tech_path = str(path_data_case_study / f"technologies/CCSTechnologies/MEA_{suitable_mea}.json")
        network_emission_flux.at[idx, 'mea_technology'] = mea_tech_path

    return network_emission_flux


def assign_ccs_technologies(network_location, network_emission_flux, path_data_case_study, input_data_path):
    """
    Assigns appropriate technologies to nodes based on their type and previously determined MEA technology.
    Handles nodes with multiple emitters by accumulating all required technologies.

    Parameters:
        - network_location: DataFrame containing node information
        - network_emission_flux: DataFrame containing emission data and MEA technology assignments
        - path_data_case_study: Path to the case study data directory
        - input_data_path: Path to the input data directory

    Returns:
        - None
    """
    # Group by unique node names to handle multiple emitters per node
    unique_nodes = network_location['node_name'].unique()

    for node_name in unique_nodes:
        # Get all rows for this node
        node_rows = network_location[network_location['node_name'] == node_name]

        # Initialize technology sets to avoid duplicates
        existing_techs_set = set()
        new_techs_set = set()

        # Process each row for this node
        for idx, row in node_rows.iterrows():
            node_type = row['node_type']

            if node_type == "Storage":
                # Storage nodes get permanent CO2 storage technology
                # NOTE: Updated to check if Sink folder exists
                storage_tech_path = path_data_case_study / "technologies/Sink/PermanentStorage_CO2_simple.json"
                if storage_tech_path.exists():
                    new_techs_set.add("PermanentStorage_CO2_simple")
                    print(f"Found storage technology at: {storage_tech_path}")
                else:
                    print(f"Warning: Storage technology file not found at {storage_tech_path}")
                    # Check if it's in the main technologies folder
                    alt_storage_path = path_data_case_study / "technologies/PermanentStorage_CO2_simple.json"
                    if alt_storage_path.exists():
                        new_techs_set.add("PermanentStorage_CO2_simple")
                        print(f"Found storage technology at alternative path: {alt_storage_path}")
                    else:
                        new_techs_set.add("PermanentStorage_CO2_simple")  # Add anyway, let system handle

            elif node_type == "Transport":
                # Transport nodes don't require specific technologies
                pass

            else:  # Emitter nodes (Waste, Cement, Other)
                # Add the MEA technology if it exists - get it from network_emission_flux
                emission_rows = network_emission_flux[network_emission_flux['node_name'] == node_name]

                # Find the emission row that matches this specific emitter type
                matching_emission_row = None
                for _, emission_row in emission_rows.iterrows():
                    if emission_row['node_type'] == node_type:
                        matching_emission_row = emission_row
                        break

                if matching_emission_row is not None:
                    mea_tech = matching_emission_row.get('mea_technology')
                    if pd.notna(mea_tech):
                        # Extract just the filename without extension from the MEA technology path
                        mea_tech_filename = Path(mea_tech).stem
                        new_techs_set.add(mea_tech_filename)

                # Assign appropriate emitter technology based on node type
                # NOTE: These should match the files in the Emitter subfolder
                if node_type == "Waste":
                    existing_techs_set.add("WasteToEnergyEmitter")
                elif node_type == "Cement":
                    existing_techs_set.add("CementEmitter")
                elif node_type == "Other":
                    existing_techs_set.add("UnspecifiedEmitter")

        # Convert sets to lists and then to dictionaries
        existing_techs = list(existing_techs_set)
        new_techs = list(new_techs_set)

        # Read the node's current Technology.json file
        tech_file_path = input_data_path / "period1" / "node_data" / node_name / "Technologies.json"

        # Convert lists to dictionary with default capacity values
        existing_techs_dict = {tech: 0.0 for tech in existing_techs}
        new_techs_dict = {tech: 0.0 for tech in new_techs}

        technologies = {
            "existing": existing_techs_dict,
            "new": new_techs_dict,
        }

        # Write updated technologies to the file
        with open(tech_file_path, "w") as json_file:
            json.dump(technologies, json_file, indent=4)

        print(
            f"Technologies assigned to {node_name}: existing={list(existing_techs_dict.keys())}, new={list(new_techs_dict.keys())}")


def update_network_distance_matrix(input_data_path, network_data_dict, network_types, decimal_places=2):
    """
    Update distance matrices for multiple network types using network data where values > 0 represent distances.

    Parameters:
    - input_data_path: Path object pointing to the input data directory
    - network_data_dict: Dictionary mapping network types to their data (values > 0 are distances, 0 means no connection)
    - network_types: List of network type folder names (e.g., ['CO2_Pipeline', 'CO2Truck', 'CO2Railway'])
    - decimal_places: Number of decimal places to round to (default: 2)
    """
    # Load the template distance CSV (empty matrix with node names)
    template_distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "distance.csv",
                                    sep=";", index_col=0)

    # Process each network type with its corresponding data
    for network_type in network_types:
        # Get the corresponding network data
        if network_type == 'CO2_Pipeline':
            network_data = network_data_dict.get('pipeline')
        elif network_type == 'CO2Truck':
            network_data = network_data_dict.get('truck')
        elif network_type == 'CO2Railway':
            network_data = network_data_dict.get('railway')
        else:
            print(f"Warning: No data mapping found for network type {network_type}")
            continue

        if network_data is None:
            print(f"Warning: No data found for network type {network_type}")
            continue

        # Create updated distance matrix
        updated_distance = template_distance.copy().astype(float)

        # Update the template matrix with distance values (keep original values where > 0, set 0 where = 0)
        updated_distance.iloc[:, :] = network_data.values

        # Round to specified decimal places
        updated_distance = updated_distance.round(decimal_places)

        # Save the updated distance matrix to the specific network type folder
        output_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "distance.csv"
        updated_distance.to_csv(output_path, sep=";")

    return True


def update_network_connection_matrix(input_data_path, network_data_dict):
    """
    Update connection matrices for multiple network types.
    Values > 0 in network data indicate connection (converted to 1), values = 0 indicate no connection.

    Parameters:
    - input_data_path: Path object pointing to the input data directory
    - network_data_dict: Dictionary with keys 'pipeline', 'truck', 'railway' containing network data
    """
    # Load the template connection CSV (empty matrix with node names)
    template_connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv",
                                      sep=";", index_col=0)

    # Mapping from network data keys to network type folders
    network_type_mapping = {
        'pipeline': 'CO2_Pipeline',
        'truck': 'CO2Truck',
        'railway': 'CO2Railway'
    }

    # Process each network data type
    for data_key, network_type in network_type_mapping.items():
        if data_key not in network_data_dict:
            print(f"Warning: {data_key} data not found in network_data_dict")
            continue

        network_data = network_data_dict[data_key]

        # Create updated connection matrix for this network type
        updated_connection = template_connection.copy()

        # Convert network data to connection matrix: values > 0 become 1, values = 0 stay 0
        connection_values = (network_data.values > 0).astype(int)
        updated_connection.iloc[:, :] = connection_values

        # Save the updated connection matrix to the specific network type folder
        output_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "connection.csv"
        updated_connection.to_csv(output_path, sep=";")

    return True


def update_network_size_max_arcs(input_data_path, network_data_dict, size_max):
    """
    Update size_max_arcs matrices for multiple network types using connection data multiplied by size_max.

    Parameters:
    - input_data_path: Path object pointing to the input data directory
    - network_data_dict: Dictionary with keys 'pipeline', 'truck', 'railway' containing network data
    - size_max: Predefined maximum size value to multiply with connection matrix
    """
    # Load the template size_max_arcs CSV (empty matrix with node names)
    template_size_max = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv",
                                    sep=";", index_col=0)

    # Mapping from network data keys to network type folders
    network_type_mapping = {
        'pipeline': 'CO2_Pipeline',
        'truck': 'CO2Truck',
        'railway': 'CO2Railway'
    }

    # Process each network data type
    for data_key, network_type in network_type_mapping.items():
        if data_key not in network_data_dict:
            print(f"Warning: {data_key} data not found in network_data_dict")
            continue

        network_data = network_data_dict[data_key]

        # Create updated size_max_arcs matrix for this network type
        updated_size_max = template_size_max.copy().astype(float)

        # Create connection matrix: values > 0 become 1, values = 0 stay 0
        connection_matrix = (network_data.values > 0).astype(int)

        # Create size_max_arcs matrix: connection_matrix * size_max
        size_max_values = connection_matrix * size_max

        # Update the template matrix with the size_max_arcs values
        updated_size_max.iloc[:, :] = size_max_values

        # Save the updated size_max_arcs matrix to the specific network type folder
        output_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "size_max_arcs.csv"
        updated_size_max.to_csv(output_path, sep=";")

    return True


def update_network_size_min_arcs(input_data_path, network_data_dict, network_emission_flux):
    """
    Update size_min_arcs matrices for multiple network types using connection data multiplied by
    the annual emission value of the start node (from node).

    Parameters:
    - input_data_path: Path object pointing to the input data directory
    - network_data_dict: Dictionary with keys 'pipeline', 'truck', 'railway' containing network data
    - network_emission_flux: DataFrame containing node emission data with 'node_name' and 'annual_emission' columns
    """
    # Ensure annual_emission column exists
    if 'annual_emission' not in network_emission_flux.columns:
        raise ValueError("'annual_emission' column not found. Please run calculate_annual_emission_values() first.")

    # Load the template size_min_arcs CSV (empty matrix with node names)
    template_size_min = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv",
                                    sep=";", index_col=0)

    # Create a mapping from node_name to annual_emission for quick lookup
    # Handle multiple emitters per node by summing their annual emissions
    node_emission_map = network_emission_flux.groupby('node_name')['annual_emission'].sum().to_dict()

    # Mapping from network data keys to network type folders
    network_type_mapping = {
        'pipeline': 'CO2_Pipeline',
        'truck': 'CO2Truck',
        'railway': 'CO2Railway'
    }

    # Process each network data type
    for data_key, network_type in network_type_mapping.items():
        if data_key not in network_data_dict:
            print(f"Warning: {data_key} data not found in network_data_dict")
            continue

        network_data = network_data_dict[data_key]

        # Create updated size_min_arcs matrix for this network type
        updated_size_min = template_size_min.copy().astype(float)

        # Get node names from template (these are the row and column indices)
        node_names = updated_size_min.index.tolist()

        # Initialize the matrix with zeros
        updated_size_min.iloc[:, :] = 0.0

        # Iterate through each cell in the matrix
        for i, from_node in enumerate(node_names):
            for j, to_node in enumerate(node_names):
                # Check if nodes are connected in the network data (values > 0 mean connection)
                connection_value = network_data.iloc[i, j]

                if connection_value > 0:  # Nodes are connected
                    # Get annual emission of the from_node (start node)
                    annual_emission = node_emission_map.get(from_node, 0)

                    if annual_emission > 0:
                        # Round to whole numbers for cleaner output
                        updated_size_min.iloc[i, j] = round(annual_emission, 0)
                else:
                    updated_size_min.iloc[i, j] = 0.0

        # Save the updated size_min_arcs matrix to the specific network type folder
        output_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "size_min_arcs.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        updated_size_min.to_csv(output_path, sep=";")

    return True


def process_gamma_sheets_to_csv(path_files_network_capex, input_data_path):
    """
    Process gamma sheets from capex_defined_per_arc.xlsx and save them as separate CSV files
    in the CO2_Pipeline folder.

    Parameters:
        - path_files_network_capex: Path to the network capex metrics directory
        - input_data_path: Path to the input data directory

    Returns:
        - gamma_data_dict: Dictionary containing the gamma data for potential future use
    """

    # Define the path to the capex file
    capex_file_path = path_files_network_capex / "capex_defined_per_arc.xlsx"

    # Check if the file exists
    if not capex_file_path.exists():
        print(f"Warning: capex_defined_per_arc.xlsx not found at {capex_file_path}")
        return None

    # Define the CO2_Pipeline output directory
    pipeline_output_dir = input_data_path / "period1" / "network_topology" / "new" / "CO2_Pipeline"
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dictionary to store gamma data
    gamma_data_dict = {}

    # List of gamma sheets to process
    gamma_sheets = ['gamma1', 'gamma2', 'gamma3', 'gamma4']

    print(f"Processing gamma sheets from {capex_file_path}...")

    # Process each gamma sheet
    for sheet_name in gamma_sheets:
        try:
            # Read the gamma sheet from Excel
            gamma_df = pd.read_excel(capex_file_path, sheet_name=sheet_name, index_col=0)

            # Store in dictionary
            gamma_data_dict[sheet_name] = gamma_df

            # Define output CSV path
            csv_output_path = pipeline_output_dir / f"{sheet_name}.csv"

            # Save as CSV with semicolon separator (consistent with other files)
            gamma_df.to_csv(csv_output_path, sep=";")

            print(f"✅ Successfully processed {sheet_name}: {gamma_df.shape} -> {csv_output_path}")

        except Exception as e:
            print(f"❌ Error processing sheet '{sheet_name}': {e}")
            continue

    # Summary
    successful_sheets = len(gamma_data_dict)
    print(f"\nGamma sheets processing completed:")
    print(f"  - Successfully processed: {successful_sheets}/{len(gamma_sheets)} sheets")
    print(f"  - Output directory: {pipeline_output_dir}")
    print(f"  - Files created: {', '.join([f'{sheet}.csv' for sheet in gamma_data_dict.keys()])}")

    return gamma_data_dict


def update_node_carbon_cost(input_data_path, carbon_tax, node_names):
    """
    Update carbon cost data for all nodes in the network.

    This function applies a uniform carbon tax to all nodes by updating their CarbonCost.csv files
    with the specified carbon tax value for all 8760 hours of the year.

    Parameters:
        - input_data_path: Path to the input data directory
        - carbon_tax: Carbon tax value to apply (e.g., 100 for €100/tonne CO2)
        - node_names: List of all node names in the network

    Returns:
        - None (updates CSV files on disk)
    """

    # Create carbon price array for all hours in a year (8760 hours)
    carbon_price = np.ones(8760) * carbon_tax

    # Counter for successful updates
    updated_nodes = 0
    failed_nodes = []

    # Process each node
    for node_name in node_names:
        try:
            # Define path to the node's CarbonCost.csv file
            carbon_cost_path = (
                    input_data_path / "period1" / "node_data" / node_name / "CarbonCost.csv"
            )

            # Check if the file exists
            if not carbon_cost_path.exists():
                print(f"Warning: CarbonCost.csv not found for node {node_name}")
                failed_nodes.append(node_name)
                continue

            # Read the carbon cost template
            carbon_cost_template = pd.read_csv(carbon_cost_path, sep=";", index_col=0, header=0)

            # Update the price column with the carbon tax
            carbon_cost_template["price"] = carbon_price

            # Reset index and save back to CSV
            carbon_cost_template = carbon_cost_template.reset_index()
            carbon_cost_template.to_csv(carbon_cost_path, sep=";", index=False)

            updated_nodes += 1

        except Exception as e:
            print(f"Error updating carbon cost for node {node_name}: {e}")
            failed_nodes.append(node_name)

    # Print summary
    print(f"Carbon cost update completed:")
    print(f"  - Successfully updated: {updated_nodes} nodes")
    print(f"  - Failed updates: {len(failed_nodes)} nodes")
    print(f"  - Carbon tax applied: ${carbon_tax}/tonne CO2")

    if failed_nodes:
        print(f"  - Failed nodes: {failed_nodes}")

    return True


def load_climate_data_from_api_robust(folder_path: str | Path, dataset: str = "JRC"):
    """
    Reads in climate data for a full year from a folder containing node data and writes it to the respective file.
    Enhanced to handle offshore nodes and other API failures gracefully.

    Parameters:
    - folder_path: Path to the folder containing node data and NodeLocations.csv
    - dataset: Dataset to import from, can be JRC (only onshore)

    Returns:
    - Tuple of (successful_nodes, failed_nodes, offshore_nodes)
    """
    # Convert to Path
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    # Import inside function to avoid import issues
    from adopt_net0.data_preprocessing.data_loading import import_jrc_climate_data

    # Read NodeLocations.csv with node column as index
    node_locations_path = os.path.join(folder_path, "NodeLocations.csv")
    node_locations_df = pd.read_csv(
        node_locations_path, sep=";", names=["node", "lon", "lat", "alt"], header=0
    )

    if node_locations_df.isnull().values.any():
        raise Exception("Please specify longitude, latitude and altitude for each node")

    # Read nodes and investment_periods from the JSON file
    json_file_path = os.path.join(folder_path, "Topology.json")
    with open(json_file_path, "r") as json_file:
        topology = json.load(json_file)

    year = (
        int(topology["start_date"].split("-")[0])
        if topology["start_date"]
        else "typical_year"
    )

    failed_nodes = []
    successful_nodes = []
    offshore_nodes = []

    for period in topology["investment_periods"]:
        for node_name in topology["nodes"]:
            # Read lon, lat, and alt for this node name from node_locations_df
            node_data = node_locations_df[node_locations_df["node"] == node_name]
            lon = node_data["lon"].values[0]
            lat = node_data["lat"].values[0]
            alt = node_data["alt"].values[0]

            if dataset == "JRC":
                try:
                    print(f"Importing Climate Data for {node_name}...")
                    # Fetch climate data for the node
                    data = import_jrc_climate_data(lon, lat, year, alt)
                    print(f"Importing Climate Data for {node_name} successful")
                    successful_nodes.append(node_name)
                except Exception as e:
                    error_msg = str(e)
                    print(f"Failed to import climate data for {node_name}: {e}")

                    # Check if it's likely an offshore location
                    if "400" in error_msg or "offshore" in error_msg.lower():
                        print(f"  -> {node_name} appears to be offshore (coordinates: {lon}, {lat})")
                        offshore_nodes.append(node_name)
                    else:
                        print(f"  -> Other API issue for {node_name}")

                    failed_nodes.append(node_name)
                    continue
            else:
                raise Exception("Other APIs are not available")

            # Write data to CSV file
            output_folder = os.path.join(folder_path, period, "node_data", node_name)
            output_file = os.path.join(output_folder, "ClimateData.csv")
            existing_data = pd.read_csv(output_file, sep=";")

            # Fill in existing data with data from the fetched DataFrame based on column names
            for column, value in data["dataframe"].items():
                existing_data[column] = value.values[: len(existing_data)]

            # Save the updated data back to ClimateData.csv
            existing_data.to_csv(output_file, index=False, sep=";")

    # Enhanced summary
    print(f"\nSummary:")
    print(f"Successfully processed {len(successful_nodes)} nodes: {successful_nodes}")
    if offshore_nodes:
        print(f"Failed to process {len(failed_nodes)} nodes: {failed_nodes}")
        print(f"\n💡 Offshore nodes detected: {offshore_nodes}")
        print(f"   This is expected behavior for offshore locations.")
        print(f"   The optimization can proceed without climate data for these nodes.")

    return successful_nodes, failed_nodes, offshore_nodes


def update_carrier_data(input_data_path, electricity_price_data, network_emission_flux,
                        path_files_technologies, node_names, co2_intensity_electricity,
                        heat_convert_factor, electricity_import_limit=100, heat_import_limit=200):
    """
    Update carrier data including import limits, import prices, import emission factors, and demands for all nodes.

    Parameters:
        - input_data_path: Path to the input data directory
        - electricity_price_data: DataFrame containing hourly electricity prices
        - network_emission_flux: DataFrame containing emission data with node_name, node_type, and annual_emission
        - path_files_technologies: Path to the technologies directory containing emission factor JSON files
        - node_names: List of all node names in the network
        - co2_intensity_electricity: CO2 emission factor for electricity (kg CO2/kWh)
        - heat_convert_factor: Factor to calculate heat price and emission factor from electricity (default: 2.6)
        - electricity_import_limit: Import limit for electricity (default: 100)
        - heat_import_limit: Import limit for heat (default: 200)

    Returns:
        - None (calls adopt.fill_carrier_data to update files)
    """

    # Import the adopt module (assuming it's already imported in main.py)
    import adopt_net0 as adopt

    # Calculate heat emission factor
    co2_intensity_heat = co2_intensity_electricity / heat_convert_factor

    # Update import limits for electricity and heat for all nodes
    adopt.fill_carrier_data(input_data_path,
                            value_or_data=electricity_import_limit,
                            columns=['Import limit'],
                            carriers=['electricity'],
                            nodes=node_names)

    adopt.fill_carrier_data(input_data_path,
                            value_or_data=heat_import_limit,
                            columns=['Import limit'],
                            carriers=['heat'],
                            nodes=node_names)

    # Update import emission factors for electricity and heat for all nodes
    adopt.fill_carrier_data(input_data_path,
                            value_or_data=co2_intensity_electricity,
                            columns=['Import emission factor'],
                            carriers=['electricity'],
                            nodes=node_names)

    adopt.fill_carrier_data(input_data_path,
                            value_or_data=co2_intensity_heat,
                            columns=['Import emission factor'],
                            carriers=['heat'],
                            nodes=node_names)

    # Process electricity pricing data
    print(f"Input electricity price data shape: {electricity_price_data.shape}")
    print(f"Columns: {electricity_price_data.columns.tolist()}")

    # Extract the price column
    if 'Day-ahead Price (EUR/MWh)' in electricity_price_data.columns:
        electricity_prices = electricity_price_data['Day-ahead Price (EUR/MWh)'].values
        price_column_name = 'Day-ahead Price (EUR/MWh)'
    else:
        # Try to find a price column with different name
        price_columns = [col for col in electricity_price_data.columns if 'price' in col.lower()]
        if price_columns:
            electricity_prices = electricity_price_data[price_columns[0]].values
            price_column_name = price_columns[0]
        else:
            raise ValueError("Could not find electricity price column in the data")

    print(f"Extracted {len(electricity_prices)} price values from column '{price_column_name}'")

    # Handle leap year data more intelligently
    if len(electricity_prices) == 8784:  # Leap year (366 days * 24 hours)
        print("Detected leap year data (8784 hours). Removing Feb 29th to get 8760 hours.")

        # Parse datetime from MTU column with improved logic
        temp_df = electricity_price_data.copy()

        if 'MTU (CET/CEST)' in temp_df.columns:
            mtu_col = temp_df['MTU (CET/CEST)']

            # Debug: Show first few MTU values to understand the format
            print(f"Sample MTU values:")
            for i in range(min(5, len(mtu_col))):
                print(f"  {i}: {mtu_col.iloc[i]}")

            # Try multiple parsing strategies
            datetime_parsed = False

            # Strategy 1: Try the actual format "dd/mm/yyyy hh:mm:ss - dd/mm/yyyy hh:mm:ss"
            try:
                # Extract the start datetime from MTU string
                start_times = mtu_col.str.split(' - ').str[0]  # Get part before " - "
                temp_df['datetime'] = pd.to_datetime(start_times, format='%d/%m/%Y %H:%M:%S', errors='coerce')

                # Check if parsing was successful
                valid_dates = temp_df['datetime'].notna().sum()
                if valid_dates > len(temp_df) * 0.9:  # At least 90% successfully parsed
                    datetime_parsed = True
                    print(
                        f"Successfully parsed MTU datetime using format dd/mm/yyyy hh:mm:ss ({valid_dates}/{len(temp_df)} values)")
                else:
                    raise ValueError(f"Format 1 failed - only {valid_dates}/{len(temp_df)} values parsed")

            except Exception as e:
                print(f"Datetime parsing strategy 1 failed: {e}")

            # Strategy 2: Try European format with dots "dd.mm.yyyy hh:mm"
            if not datetime_parsed:
                try:
                    start_times = mtu_col.str.split(' - ').str[0]
                    temp_df['datetime'] = pd.to_datetime(start_times, format='%d.%m.%Y %H:%M', errors='coerce')

                    valid_dates = temp_df['datetime'].notna().sum()
                    if valid_dates > len(temp_df) * 0.9:
                        datetime_parsed = True
                        print(
                            f"Successfully parsed MTU datetime using format dd.mm.yyyy hh:mm ({valid_dates}/{len(temp_df)} values)")
                    else:
                        raise ValueError(f"Format 2 failed - only {valid_dates}/{len(temp_df)} values parsed")

                except Exception as e:
                    print(f"Datetime parsing strategy 2 failed: {e}")

            # Strategy 3: Try automatic pandas parsing
            if not datetime_parsed:
                try:
                    start_times = mtu_col.str.split(' - ').str[0]
                    temp_df['datetime'] = pd.to_datetime(start_times, errors='coerce')

                    valid_dates = temp_df['datetime'].notna().sum()
                    if valid_dates > len(temp_df) * 0.9:
                        datetime_parsed = True
                        print(
                            f"Successfully parsed MTU datetime using automatic pandas parsing ({valid_dates}/{len(temp_df)} values)")
                    else:
                        raise ValueError(f"Format 3 failed - only {valid_dates}/{len(temp_df)} values parsed")

                except Exception as e:
                    print(f"Datetime parsing strategy 3 failed: {e}")

            # Strategy 4: Try parsing with regex for forward slash format
            if not datetime_parsed:
                try:
                    # Extract date and time parts for forward slash format
                    date_part = mtu_col.str.extract(r'(\d{2}/\d{2}/\d{4})')[0]
                    time_part = mtu_col.str.extract(r'(\d{2}:\d{2}:\d{2})')[0]

                    datetime_str = date_part + ' ' + time_part
                    temp_df['datetime'] = pd.to_datetime(datetime_str, format='%d/%m/%Y %H:%M:%S', errors='coerce')

                    valid_dates = temp_df['datetime'].notna().sum()
                    if valid_dates > len(temp_df) * 0.9:
                        datetime_parsed = True
                        print(
                            f"Successfully parsed MTU datetime using regex extraction for forward slash format ({valid_dates}/{len(temp_df)} values)")
                    else:
                        raise ValueError(f"Format 4 failed - only {valid_dates}/{len(temp_df)} values parsed")

                except Exception as e:
                    print(f"Datetime parsing strategy 4 failed: {e}")

            # Strategy 5: Create sequential datetime assuming hourly data starting from 2024-01-01
            if not datetime_parsed:
                print("All datetime parsing strategies failed. Creating sequential datetime index.")
                start_date = datetime(2024, 1, 1)
                temp_df['datetime'] = pd.date_range(start=start_date, periods=len(temp_df), freq='H')
                datetime_parsed = True

        else:
            # Create datetime index assuming data starts from January 1st, 2024
            print("No MTU column found. Creating datetime index assuming hourly data starting from 2024-01-01")
            start_date = datetime(2024, 1, 1)
            temp_df['datetime'] = pd.date_range(start=start_date, periods=len(temp_df), freq='H')

        # Debug: Check what year and date range we actually have
        if 'datetime' in temp_df.columns:
            min_date = temp_df['datetime'].min()
            max_date = temp_df['datetime'].max()
            print(f"Date range in data: {min_date} to {max_date}")

            # Check for February 29th specifically
            feb29_rows = temp_df[(temp_df['datetime'].dt.month == 2) & (temp_df['datetime'].dt.day == 29)]
            print(f"Found {len(feb29_rows)} rows with February 29th data")

            if len(feb29_rows) > 0:
                print(f"Feb 29th date range: {feb29_rows['datetime'].min()} to {feb29_rows['datetime'].max()}")

        # Now remove February 29th data (24 hours)
        rows_before = len(temp_df)
        temp_df = temp_df[~((temp_df['datetime'].dt.month == 2) & (temp_df['datetime'].dt.day == 29))]
        rows_after = len(temp_df)

        print(f"Removed {rows_before - rows_after} rows for February 29th")

        # If no rows were removed, try a different approach - just take the first 8760 values
        if rows_before == rows_after and len(electricity_prices) == 8784:
            print("No Feb 29th rows found to remove. Using first 8760 values instead.")
            electricity_prices = electricity_prices[:8760]
        else:
            # Extract prices after removing Feb 29th
            electricity_prices = temp_df[price_column_name].values

        # Verify we now have the correct number of values
        print(f"After processing leap year data: {len(electricity_prices)} values")

    elif len(electricity_prices) > 8760 and len(electricity_prices) != 8784:
        print(
            f"Warning: Expected 8760 or 8784 hourly values, got {len(electricity_prices)}. Truncating to first 8760 values.")
        electricity_prices = electricity_prices[:8760]
    elif len(electricity_prices) < 8760:
        print(
            f"Warning: Expected 8760 hourly values, got {len(electricity_prices)}. Padding with the last available value.")
        # Pad with the last available value
        electricity_prices = np.pad(electricity_prices,
                                    (0, 8760 - len(electricity_prices)),
                                    'edge')

    # Final check to ensure we have exactly 8760 values
    if len(electricity_prices) != 8760:
        raise ValueError(f"After processing, expected exactly 8760 values but got {len(electricity_prices)}")

    # Calculate heat prices (electricity_price / heat_convert_factor)
    heat_prices = electricity_prices / heat_convert_factor

    # Update import prices for electricity and heat for all nodes
    adopt.fill_carrier_data(input_data_path,
                            value_or_data=electricity_prices,
                            columns=['Import price'],
                            carriers=['electricity'],
                            nodes=node_names)

    adopt.fill_carrier_data(input_data_path,
                            value_or_data=heat_prices,
                            columns=['Import price'],
                            carriers=['heat'],
                            nodes=node_names)

    # Load emission factors from technology JSON files
    emission_factors = {}

    # Define mapping from node_type to technology file and emission factor key path
    # NOTE: Updated paths to include the Emitter subfolder
    node_type_mapping = {
        'Waste': ('Emitter/WasteToEnergyEmitter.json', ['Performance', 'emission_factor']),
        'Cement': ('Emitter/CementEmitter.json', ['Performance', 'emission_factor']),
        'Other': ('Emitter/UnspecifiedEmitter.json', ['Performance', 'emission_factor'])
    }

    # Load emission factors from JSON files
    for node_type, (filename, factor_key_path) in node_type_mapping.items():
        tech_file_path = path_files_technologies / filename  # Now includes Emitter/ subfolder
        if tech_file_path.exists():
            with open(tech_file_path, 'r') as f:
                tech_data = json.load(f)

                # Navigate through the nested structure
                try:
                    current_data = tech_data
                    for key in factor_key_path:
                        current_data = current_data[key]

                    emission_factors[node_type] = current_data
                    print(f"✅ Loaded emission factor for {node_type}: {current_data}")
                except KeyError as e:
                    print(f"Warning: Key path {factor_key_path} not found in {filename} (missing: {e})")
                    emission_factors[node_type] = 1.0  # Default value
        else:
            print(f"Warning: Technology file {filename} not found at {tech_file_path}")
            emission_factors[node_type] = 1.0  # Default value

    # Process demands for each node based on emission data
    # Handle multiple emitters per node by accumulating demands by carrier type
    node_demands = {}
    emitter_count_by_node = {}

    for _, row in network_emission_flux.iterrows():
        node_name = row['node_name']
        node_type = row['node_type']

        # Use the pre-calculated annual_emission value
        annual_emission = row['annual_emission']

        # Skip non-emitter nodes or nodes with zero emissions
        if node_type in ['Storage', 'Transport'] or annual_emission == 0:
            continue

        # Count emitters per node for logging
        if node_name not in emitter_count_by_node:
            emitter_count_by_node[node_name] = []
        emitter_count_by_node[node_name].append(f"{node_type}({annual_emission})")

        if node_type in emission_factors:
            # Calculate demand using emission factor
            demand_value = annual_emission * emission_factors[node_type]

            # Create carrier name based on node type
            if node_type == 'Waste':
                carrier_name = 'waste'
            elif node_type == 'Cement':
                carrier_name = 'cement'
            elif node_type == 'Other':
                carrier_name = 'industrial_product'
            else:
                carrier_name = node_type.lower()

            # Store demand data (accumulate if multiple emitters of same type per node)
            if node_name not in node_demands:
                node_demands[node_name] = {}

            if carrier_name not in node_demands[node_name]:
                node_demands[node_name][carrier_name] = 0

            node_demands[node_name][carrier_name] += demand_value

    # Log nodes with multiple emitters
    multi_emitter_nodes = {node: emitters for node, emitters in emitter_count_by_node.items() if len(emitters) > 1}
    if multi_emitter_nodes:
        print(f"Nodes with multiple emitters detected:")
        for node_name, emitters in multi_emitter_nodes.items():
            print(f"  {node_name}: {', '.join(emitters)}")
            if node_name in node_demands:
                print(f"    Total demands: {node_demands[node_name]}")

    # Update demands using adopt.fill_carrier_data
    for node_name, carriers_demands in node_demands.items():
        for carrier_name, demand_value in carriers_demands.items():
            adopt.fill_carrier_data(input_data_path,
                                    value_or_data=demand_value,
                                    columns=['Demand'],
                                    carriers=[carrier_name],
                                    nodes=[node_name])

    # Print summary
    print(f"Carrier data update completed:")
    print(f"  - Electricity import limit: {electricity_import_limit} for all nodes")
    print(f"  - Heat import limit: {heat_import_limit} for all nodes")
    print(f"  - Electricity emission factor: {co2_intensity_electricity} kg CO2/kWh for all nodes")
    print(f"  - Heat emission factor: {co2_intensity_heat:.4f} kg CO2/kWh for all nodes")
    print(f"  - Electricity prices: {len(electricity_prices)} hourly values applied to all nodes")
    print(f"  - Heat prices: derived from electricity prices (electricity_price / {heat_convert_factor})")
    print(f"  - Demands updated for {len(node_demands)} nodes with emissions")
    print(f"  - Total emitters processed: {sum(len(emitters) for emitters in emitter_count_by_node.values())}")
    print(f"  - Emission factors used: {emission_factors}")

    # Show detailed demand summary
    if node_demands:
        print(f"\nDetailed demand summary by node:")
        for node_name, carriers_demands in node_demands.items():
            total_node_demand = sum(carriers_demands.values())
            carrier_details = ', '.join([f"{carrier}: {demand:.0f}" for carrier, demand in carriers_demands.items()])
            print(f"  {node_name}: {carrier_details} (Total: {total_node_demand:.0f})")

    return True