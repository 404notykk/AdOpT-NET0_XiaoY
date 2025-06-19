import pandas as pd
import json
import os
from pathlib import Path


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


def calculate_production_profiles(network_emission_flux, path_data_case_study):
    """
    Calculate production profiles by back-calculating from emission profiles using emission factors from emitter JSON files.

    This function reads emission factors from emitter technology JSON files and back-calculates
    production profiles using the formula: Production = Emission / Emission_Factor

    Parameters:
        - network_emission_flux: DataFrame containing emission data for each node with 'annual_emission' column
        - path_data_case_study: Path to the case study data directory containing emitter JSON files

    Returns:
        - network_emission_flux: Updated DataFrame with 'annual_production' column added
    """
    # Ensure annual_emission column exists
    if 'annual_emission' not in network_emission_flux.columns:
        raise ValueError("'annual_emission' column not found. Please run calculate_annual_emission_values() first.")

    # Initialize production profiles DataFrame
    production_profiles = network_emission_flux.copy()

    # Path to emitter JSON files
    emitter_files_path = path_data_case_study / "technologies"

    # Get all emitter JSON files (look for files containing 'emitter' in the name)
    emitter_files = []
    for root, dirs, files in os.walk(emitter_files_path):
        for file in files:
            if 'emitter' in file.lower() and file.endswith('.json'):
                emitter_files.append(Path(root) / file)

    # Dictionary to store emission factors by emitter type
    emission_factors = {}

    # Read emission factors from JSON files
    for emitter_file in emitter_files:
        try:
            with open(emitter_file, 'r') as f:
                emitter_data = json.load(f)

            # Extract emitter name/type from filename
            emitter_type = emitter_file.stem.lower()

            # Look for emission factor in the JSON structure
            emission_factor = None
            if 'emission_factor' in emitter_data:
                emission_factor = emitter_data['emission_factor']
            elif 'parameters' in emitter_data and 'emission_factor' in emitter_data['parameters']:
                emission_factor = emitter_data['parameters']['emission_factor']
            elif 'techno_economic' in emitter_data and 'emission_factor' in emitter_data['techno_economic']:
                emission_factor = emitter_data['techno_economic']['emission_factor']
            elif 'co2_emission_factor' in emitter_data:
                emission_factor = emitter_data['co2_emission_factor']

            if emission_factor is not None:
                emission_factors[emitter_type] = emission_factor
            else:
                print(f"Warning: Could not find emission factor for {emitter_type}")

        except Exception as e:
            print(f"Warning: Error reading {emitter_file}: {e}")

    # Calculate production profiles for each node
    for idx, row in production_profiles.iterrows():
        node_name = row['node_name']
        node_type = row.get('node_type', '').lower()

        # Skip non-emitter nodes (Storage and Transport)
        if node_type in ["storage", "transport"]:
            production_profiles.at[idx, 'annual_production'] = 0
            continue

        # Map node type to emitter type for emission factor lookup
        emitter_type_mapping = {
            'waste': 'wastetoenergyemitter',
            'cement': 'cementemitter',
            'other': 'unspecifiedemitter'
        }

        emitter_type = emitter_type_mapping.get(node_type, 'unspecifiedemitter')

        # Get emission factor for this emitter type
        emission_factor = None
        for key, value in emission_factors.items():
            if emitter_type in key or node_type in key:
                emission_factor = value
                break

        # Default emission factors if not found in JSON files (kg CO2 / unit production)
        if emission_factor is None:
            default_factors = {
                'waste': 0.5,     # kg CO2 per kg waste processed
                'cement': 0.85,   # kg CO2 per kg cement produced
                'other': 0.3      # kg CO2 per unit production ???
            }
            emission_factor = default_factors.get(node_type, 0.3)
            print(f"Warning: Using default emission factor {emission_factor} for node {node_name} ({node_type})")

        # Back-calculate production from emission
        annual_emission = row.get('annual_emission', 0)
        if emission_factor > 0 and annual_emission > 0:
            annual_production = annual_emission / emission_factor
            production_profiles.at[idx, 'annual_production'] = annual_production
        else:
            production_profiles.at[idx, 'annual_production'] = 0
            if annual_emission > 0:
                print(f"Warning: Could not calculate production for node {node_name} - emission factor is {emission_factor}")

    return production_profiles


def create_demand_profiles(production_profiles, input_data_path, carriers=['electricity', 'heat']):
    """
    Create demand profile files for the optimization model based on production profiles.

    This function converts production profiles into hourly demand profiles for different energy carriers
    that the optimization model can use to drive the energy system.

    Parameters:
        - production_profiles: DataFrame containing production profiles with 'annual_production' column
        - input_data_path: Path to input data directory where demand files will be saved
        - carriers: List of energy carriers that have demand (default: ['electricity', 'heat'])

    Returns:
        - None (saves CSV files to disk)
    """
    # Ensure annual_production column exists
    if 'annual_production' not in production_profiles.columns:
        raise ValueError("'annual_production' column not found. Please run calculate_production_profiles() first.")

    # Create demand profiles for each carrier
    for carrier in carriers:
        demand_data = []

        # Process each node with production
        for idx, row in production_profiles.iterrows():
            node_name = row['node_name']
            node_type = row.get('node_type', '').lower()
            annual_production = row.get('annual_production', 0)

            # Skip nodes without production or non-emitter nodes
            if annual_production <= 0 or node_type in ["storage", "transport"]:
                continue

            # Define energy intensity factors (energy demand per unit production)
            # These factors represent how much energy is needed per unit of production
            energy_intensity = {
                'electricity': {
                    'waste': 0.8,    # kWh electricity per kg waste processed
                    'cement': 0.6,   # kWh electricity per kg cement produced
                    'other': 0.4     # kWh electricity per unit production
                },
                'heat': {
                    'waste': 1.2,    # kWh heat per kg waste processed
                    'cement': 2.5,   # kWh heat per kg cement produced
                    'other': 0.8     # kWh heat per unit production
                }
            }

            # Get energy intensity for this node type and carrier
            intensity = energy_intensity.get(carrier, {}).get(node_type, 0)

            if intensity > 0:
                # Calculate annual energy demand
                annual_energy_demand = annual_production * intensity

                # Convert to hourly demand (assuming constant demand throughout the year)
                hourly_demand = annual_energy_demand / 8760  # 8760 hours per year

                # Create hourly profile for the year
                for hour in range(1, 8761):  # 1 to 8760
                    demand_data.append({
                        'node': node_name,
                        'carrier': carrier,
                        'hour': hour,
                        'demand': round(hourly_demand, 6)
                    })

        # Convert to DataFrame and save
        if demand_data:
            demand_df = pd.DataFrame(demand_data)

            # Ensure the period1 directory exists
            period_path = input_data_path / "period1"
            period_path.mkdir(parents=True, exist_ok=True)

            # Save to CSV file
            demand_file_path = period_path / f"{carrier}_demand.csv"
            demand_df.to_csv(demand_file_path, index=False)

            print(f"Created demand profile for {carrier}: {demand_file_path}")
            print(f"Total nodes with {carrier} demand: {len(demand_df['node'].unique())}")
        else:
            print(f"Warning: No demand data generated for carrier {carrier}")


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
                new_techs_set.add("PermanentStorage_CO2_simple")

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


def update_network_distance_matrix(input_data_path, network_distance, network_types, decimal_places=2):
    """
    Update distance matrices for multiple network types using the same distance data.

    Parameters:
    - input_data_path: Path object pointing to the input data directory
    - network_distance: DataFrame containing the distance matrix from Excel (indexed by node_id)
    - network_types: List of network type folder names (e.g., ['CO2_Pipeline', 'CO2Truck', 'CO2Railway'])
    - decimal_places: Number of decimal places to round to (default: 2)
    """
    # Load the template distance CSV (empty matrix with node names)
    template_distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "distance.csv",
                                    sep=";", index_col=0)

    # Create updated distance matrix
    updated_distance = template_distance.copy().astype(float)

    # Update the template matrix with the distance values (positional copying)
    updated_distance.iloc[:, :] = network_distance.values

    # Round to specified decimal places
    updated_distance = updated_distance.round(decimal_places)

    # Save the updated distance matrix to each network type folder
    for network_type in network_types:
        output_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "distance.csv"
        updated_distance.to_csv(output_path, sep=";")

    return updated_distance


def update_network_connection_matrix(input_data_path, connection_data_dict):
    """
    Update connection matrices for multiple network types using different connection data for each type.

    Parameters:
    - input_data_path: Path object pointing to the input data directory
    - connection_data_dict: Dictionary mapping network types to their connection data
    """
    # Load the template connection CSV (empty matrix with node names)
    template_connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv",
                                      sep=";", index_col=0)

    # Process each network type with its corresponding connection data
    for network_type, connection_data in connection_data_dict.items():
        # Create updated connection matrix for this network type
        updated_connection = template_connection.copy()

        # Update the template matrix with the connection values (positional copying)
        updated_connection.iloc[:, :] = connection_data.values

        # Convert to integer type since connections are binary (0 or 1)
        updated_connection = updated_connection.astype(int)

        # Save the updated connection matrix to the specific network type folder
        output_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "connection.csv"
        updated_connection.to_csv(output_path, sep=";")

    return True


def update_network_size_max_arcs(input_data_path, connection_data_dict, size_max):
    """
    Update size_max_arcs matrices for multiple network types using connection data multiplied by size_max.

    Parameters:
    - input_data_path: Path object pointing to the input data directory
    - connection_data_dict: Dictionary mapping network types to their connection data
    - size_max: Predefined maximum size value to multiply with connection matrix
    """
    # Load the template size_max_arcs CSV (empty matrix with node names)
    template_size_max = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv",
                                    sep=";", index_col=0)

    # Process each network type with its corresponding connection data
    for network_type, connection_data in connection_data_dict.items():
        # Create updated size_max_arcs matrix for this network type
        updated_size_max = template_size_max.copy().astype(float)

        # Create size_max_arcs matrix: connection_matrix * size_max
        size_max_values = connection_data.values * size_max

        # Update the template matrix with the size_max_arcs values
        updated_size_max.iloc[:, :] = size_max_values

        # Save the updated size_max_arcs matrix to the specific network type folder
        output_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "size_max_arcs.csv"
        updated_size_max.to_csv(output_path, sep=";")

    return True


def update_network_size_min_arcs(input_data_path, connection_data_dict, network_emission_flux):
    """
    Update size_min_arcs matrices for multiple network types using connection data multiplied by
    the annual emission value of the start node (from node).

    Parameters:
    - input_data_path: Path object pointing to the input data directory
    - connection_data_dict: Dictionary mapping network types to their connection data
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

    # Process each network type with its corresponding connection data
    for network_type, connection_data in connection_data_dict.items():
        # Create updated size_min_arcs matrix for this network type
        updated_size_min = template_size_min.copy().astype(float)

        # Get node names from template (these are the row and column indices)
        node_names = updated_size_min.index.tolist()

        # Initialize the matrix with zeros
        updated_size_min.iloc[:, :] = 0.0

        # Iterate through each cell in the matrix
        for i, from_node in enumerate(node_names):
            for j, to_node in enumerate(node_names):
                # Check if nodes are connected in the connection matrix
                connection_value = connection_data.iloc[i, j]

                if connection_value == 1:  # Nodes are connected
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