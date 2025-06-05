import pandas as pd
import json
import os
from pathlib import Path


def assign_mea_technology(network_emission_flux, path_data_case_study):
    """
    Determines appropriate MEA (Monoethanolamine) carbon capture technology scale
    for emitter nodes based on their annual CO2 emissions.

    This function analyzes emission data for each node and determines the appropriate
    MEA technology scale (small, medium, large), adding it to a new column 'mea_technology'.

    Parameters:
        - network_emission_flux: DataFrame containing node information and emission data
        - path_data_case_study: Path to the case study data directory

    Returns:
        - network_emission_flux: Updated DataFrame with mea_technology column added
    """
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

    # Process each node in the network
    for node_name, row in network_emission_flux.iterrows():
        node = node_name  # Node name is the index
        node_type = row['node_type']

        # Skip non-emitter nodes (Storage and Transport)
        if node_type in ["Storage", "Transport"]:
            continue

        # Get the node's annual CO2 emission flux (kg/year)
        annual_flux = row["annual_flux"]

        # Determine CO2 concentration based on emitter type
        # (Waste emitters have 7% CO2, others have 20%)
        if node_type in ["Waste"]:
            co2_concentration = 0.07
        else:
            co2_concentration = 0.20 ??? 0.15

        # Calculate CO2 ranges for each MEA scale based on technology specs
        # Convert MEA scale from t/h to kg/year for comparison
        # 1 t/h = 1000 kg/h = 1000 * 24 * 365 kg/year = 8,760,000 kg/year
        conversion_factor = 1000 * 24 * 365  # t/h to kg/year

        mea_ranges = {}
        for scale, data in mea_data.items():
            min_co2 = co2_concentration * data["size_min"] * conversion_factor
            max_co2 = co2_concentration * data["size_max"] * conversion_factor
            mea_ranges[scale] = (min_co2, max_co2)

        # Find the MEA scale that matches the node's emission range
        suitable_mea = None
        for scale, (min_co2, max_co2) in mea_ranges.items():
            if min_co2 <= annual_flux <= max_co2:
                suitable_mea = scale
                break

        # If no exact match found, print the node and choose the closest scale
        if suitable_mea is None:
            print(
                f"Node {node} with annual_flux {annual_flux} kg/year has no exact MEA scale match. Finding closest scale...")
            distances = {}
            for scale, (min_co2, max_co2) in mea_ranges.items():
                if annual_flux < min_co2:
                    distances[scale] = min_co2 - annual_flux
                elif annual_flux > max_co2:
                    distances[scale] = annual_flux - max_co2

            suitable_mea = min(distances, key=distances.get)
            print(f"  → Assigned {suitable_mea} scale to node {node} (closest match)")

        # Store the suitable MEA technology in the mea_technology column
        mea_tech_path = str(path_data_case_study / f"technologies/CCSTechnologies/MEA_{suitable_mea}.json")
        network_emission_flux.at[node_name, 'mea_technology'] = mea_tech_path

    return network_emission_flux


def assign_ccs_technologies(network_location, network_emission_flux, path_data_case_study, input_data_path):
    """
    Assigns appropriate technologies to nodes based on their type and previously determined MEA technology.

    This function determines which technologies should be assigned to each node
    in the network, categorizing them as either existing or new technologies,
    and updates the node's technology files accordingly.

    Parameters:
        - network_location: DataFrame containing node information
        - network_emission_flux: DataFrame containing emission data and MEA technology assignments
        - path_data_case_study: Path to the case study data directory
        - input_data_path: Path to the input data directory

    Returns:
        - None
    """
    # Define paths to technology folders
    path_emitter = path_data_case_study / "technologies" / "Emitter"
    path_sink = path_data_case_study / "technologies" / "Sink"
    path_ccs = path_data_case_study / "technologies" / "CCSTechnologies"

    # Process each node in the network
    for node_name, row in network_location.iterrows():
        node_name = row["node_name"]
        node_type = row['node_type']

        # Initialize technology lists
        existing_techs = []
        new_techs = []

        if node_type == "Storage":
            # Storage nodes get permanent CO2 storage technology
            new_techs.append("PermanentStorage_CO2_simple")
        elif node_type == "Transport":
            # Transport nodes don't require specific technologies
            pass
        else:
            # All other nodes (i.e., emitters) get CO2 compressor technology
            new_techs.append("CO2_Compressor")

            # Add the MEA technology if it exists - get it from network_emission_flux
            # Find the corresponding row in network_emission_flux using node_name
            emission_row = network_emission_flux[network_emission_flux['node_name'] == node_name]
            if not emission_row.empty:
                mea_tech = emission_row.iloc[0].get('mea_technology')
                if pd.notna(mea_tech):
                    # Extract just the filename without extension from the MEA technology path
                    mea_tech_filename = Path(mea_tech).stem
                    new_techs.append(mea_tech_filename)
                    print(f"Added MEA technology {mea_tech_filename} to node {node_name}")
            else:
                print(f"Warning: Node {node_name} not found in network_emission_flux")

            # Assign appropriate emitter technology based on node type
            if node_type == "Waste":
                existing_techs.append("WasteToEnergyEmitter")
            elif node_type == "Waste_Cement":
                existing_techs.append("WasteToEnergyEmitter")
                existing_techs.append("CementEmitter")
            elif node_type == "Cement":
                existing_techs.append("CementEmitter")
            elif node_type == "Other":
                existing_techs.append("UnspecifiedEmitter")

        # Read the node's current Technology.json file
        tech_file_path = input_data_path / "period1" / "node_data" / node_name / "Technologies.json"
        with open(tech_file_path, "r") as json_file:
            technologies = json.load(json_file)

        # Update with existing and new technologies
        # Convert existing_techs list to dictionary with default capacity values
        existing_techs_dict = {tech: 0.0 for tech in existing_techs}

        technologies = {
            "existing": existing_techs_dict,
            "new": new_techs
        }

        # Write updated technologies back to the file
        with open(tech_file_path, "w") as json_file:
            json.dump(technologies, json_file, indent=4)

        print(f"Updated technologies for node {node_name}: existing={existing_techs}, new={new_techs}")