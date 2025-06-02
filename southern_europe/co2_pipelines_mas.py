"""
CO2 Pipeline Cost Model with Geographical Factors

This module extends the CO2Chain_Oeuvray class to include geographical cost factors
based on morphological features, soil type, and anthropisation.

The formula used is: co2_pipelines_mas = (1+k) * co2_pipelines_oeuvray

where k is the geographical cost factor calculated as:
k_t = \sum_{i,j} (p_{i,j} * q_{i,j,t})

with:
- p_{i,j} = x_{i,j} / \sum_{j} x_{i,j} (proportion of each geographical feature)
- q_{i,j,t} are the cost coefficients for each timeframe
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt

# Import the CO2Chain_Oeuvray class
try:
    # Try to import directly (works if this script is in the correct location)
    from database.components.networks.utilities.co2_pipelines_oeuvray import CO2Chain_Oeuvray
except ImportError:
    # Try to find the module in parent directories
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Try to find co2_pipelines_oeuvray.py in any parent directory
    found = False
    for _ in range(5):  # Try up to 5 levels up
        potential_paths = list(current_dir.glob("**/co2_pipelines_oeuvray.py"))
        if potential_paths:
            module_dir = os.path.dirname(potential_paths[0])
            sys.path.append(str(module_dir))
            found = True
            break
        current_dir = current_dir.parent

    if found:
        try:
            from co2_pipelines_oeuvray import CO2Chain_Oeuvray
        except ImportError:
            print("Module found but could not be imported. Check for circular imports or other issues.")
            raise
    else:
        raise ImportError("Could not find co2_pipelines_oeuvray.py. Please ensure it exists in the correct path.")


class GeographicalCostFactors:
    """
    Class to calculate and manage geographical cost factors for CO2 pipelines.
    The cost factors are based on morphological features, soil type, and anthropisation.
    """

    def __init__(self, data_dir=None):
        """
        Initialize the GeographicalCostFactors class.

        Args:
            data_dir (str, optional): Directory containing the geographical data files.
                If None, tries to find the files in common locations.
        """
        self.data_dir = self._find_data_dir(data_dir)
        print(f"Using data directory: {self.data_dir}")

        # Load the cost factor coefficients for different timeframes
        self.cost_coefficients = self._load_cost_coefficients()

        # Load geographical data
        self.morphological_data = None
        self.soil_data = None
        self.anthropisation_data = None

        self._load_geographical_data()

    def _find_data_dir(self, data_dir=None):
        """
        Find the directory containing the geographical data files.
        Tries several common locations if data_dir is not specified.

        Args:
            data_dir (str, optional): Directory containing the geographical data files.

        Returns:
            pathlib.Path: Path to the directory containing the data files.
        """
        if data_dir:
            return Path(data_dir)

        # Try to find the data files in common locations
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        # List of possible locations (based on the file structure in the screenshot)
        possible_locations = [
            current_dir,  # Current directory
            current_dir / "southern_europe",  # southern_europe subfolder
            Path("southern_europe"),  # southern_europe relative to working directory
            current_dir.parent / "southern_europe",  # Parent dir's southern_europe subfolder
        ]

        # Also try searching parent directories up to 5 levels
        parent_dir = current_dir
        for _ in range(5):
            parent_dir = parent_dir.parent
            possible_locations.append(parent_dir)
            possible_locations.append(parent_dir / "southern_europe")

        # Check each location for the required files
        for location in possible_locations:
            if all((location / file).exists() for file in [
                "morphological_feature_italy.csv",
                "soil_italy.csv",
                "anthropisation_italy.csv"
            ]):
                return location

        # If we can't find the files, default to the current directory
        print("Warning: Could not find the required CSV files in common locations.")
        print("Please make sure the CSV files are in the correct directory.")
        return current_dir

    def _load_cost_coefficients(self):
        """
        Load the cost coefficients from Table 1b.

        Returns:
            dict: Dictionary containing cost coefficients for different timeframes.
        """
        # Define cost coefficients from Table 1b
        coefficients = {
            'short-term': {
                'k1': {  # Geomorphology
                    'Plain': 0.0250,
                    'Hill': 0.1800,
                    'Mountain': 0.2700
                },
                'k2': {  # Soil Type
                    'Non-Rock': 0.0250,
                    'Rock': 0.6200
                },
                'k3': {  # Anthropisation
                    'Non-Anthropised': 0.0025,
                    'Anthropised': 1.1300
                }
            },
            'medium-term': {
                'k1': {  # Geomorphology
                    'Plain': 0.0250,
                    'Hill': 0.1050,
                    'Mountain': 0.1575
                },
                'k2': {  # Soil Type
                    'Non-Rock': 0.0250,
                    'Rock': 0.3525
                },
                'k3': {  # Anthropisation
                    'Non-Anthropised': 0.0025,
                    'Anthropised': 0.6475
                }
            },
            'long-term': {
                'k1': {  # Geomorphology
                    'Plain': 0.0250,
                    'Hill': 0.0554,
                    'Mountain': 0.0792
                },
                'k2': {  # Soil Type
                    'Non-Rock': 0.0250,
                    'Rock': 0.1792
                },
                'k3': {  # Anthropisation
                    'Non-Anthropised': 0.0025,
                    'Anthropised': 0.3283
                }
            }
        }

        return coefficients

    def _load_geographical_data(self):
        """
        Load geographical data from CSV files.
        """
        try:
            # Try to load the morphological data
            morph_path = self.data_dir / "morphological_feature_italy.csv"
            self.morphological_data = pd.read_csv(morph_path)
            print(f"Successfully loaded morphological data with {len(self.morphological_data)} rows")

            # Try to load the soil data
            soil_path = self.data_dir / "soil_italy.csv"
            self.soil_data = pd.read_csv(soil_path)
            print(f"Successfully loaded soil data with {len(self.soil_data)} rows")

            # Try to load the anthropisation data
            anthro_path = self.data_dir / "anthropisation_grids_italy.csv"
            self.anthropisation_data = pd.read_csv(anthro_path)
            print(f"Successfully loaded anthropisation data with {len(self.anthropisation_data)} rows")

        except FileNotFoundError as e:
            print(f"Error loading geographical data: {e}")
            print("Error: CSV files must exist with the correct names in the specified directory.")
            print("Please ensure that the following files exist:")
            print(f"  - {morph_path}")
            print(f"  - {soil_path}")
            print(f"  - {anthro_path}")
            raise e

    def calculate_p_values(self, segment_id=None):
        """
        Calculate p values for each geographical factor category.

        The formula is: p_{i,j} = x_{i,j} / sum_{j} x_{i,j}
        where x_{i,j} are the values in the geographical data files.

        Args:
            segment_id (int, optional): ID of the pipeline segment to calculate p values for.
                If None, calculates average p values across all segments.

        Returns:
            dict: Dictionary containing p values for each geographical factor category.
        """
        p_values = {
            'k1': {},  # Geomorphology
            'k2': {},  # Soil Type
            'k3': {}  # Anthropisation
        }

        # If a specific segment is provided, filter the data
        if segment_id is not None:
            # Check if the specified segment exists in all datasets
            morph_segments = set(self.morphological_data['OID'].unique())
            soil_segments = set(self.soil_data['OID'].unique())
            anthro_segments = set(self.anthropisation_data['OID'].unique())

            if segment_id not in morph_segments:
                print(f"Warning: Segment ID {segment_id} not found in morphological data.")
                print(f"Available segment IDs: {sorted(list(morph_segments))[:10]}...")
                return None

            if segment_id not in soil_segments:
                print(f"Warning: Segment ID {segment_id} not found in soil data.")
                print(f"Available segment IDs: {sorted(list(soil_segments))[:10]}...")
                return None

            if segment_id not in anthro_segments:
                print(f"Warning: Segment ID {segment_id} not found in anthropisation data.")
                print(f"Available segment IDs: {sorted(list(anthro_segments))[:10]}...")
                return None

            # Filter data for the specified segment
            morph_data = self.morphological_data[self.morphological_data['OID'] == segment_id]
            soil_data = self.soil_data[self.soil_data['OID'] == segment_id]
            anthro_data = self.anthropisation_data[self.anthropisation_data['OID'] == segment_id]
        else:
            # Use all data for average calculations
            morph_data = self.morphological_data
            soil_data = self.soil_data
            anthro_data = self.anthropisation_data

        # Verify column names in the morphological data
        morph_columns = self.morphological_data.columns
        if 'PLAIN' not in morph_columns or 'HILL' not in morph_columns or 'MOUNTAIN' not in morph_columns:
            print(f"Warning: Expected columns not found in morphological data. Found: {morph_columns}")
            print("Using default morphological proportions instead.")
            p_values['k1']['Plain'] = 0.6
            p_values['k1']['Hill'] = 0.3
            p_values['k1']['Mountain'] = 0.1
        else:
            # Calculate p values for morphological features (k1)
            morph_sum = morph_data['PLAIN'].sum() + morph_data['HILL'].sum() + morph_data['MOUNTAIN'].sum()

            if morph_sum > 0:
                p_values['k1']['Plain'] = morph_data['PLAIN'].sum() / morph_sum
                p_values['k1']['Hill'] = morph_data['HILL'].sum() / morph_sum
                p_values['k1']['Mountain'] = morph_data['MOUNTAIN'].sum() / morph_sum
            else:
                # Default values if sum is zero
                p_values['k1']['Plain'] = 0.6
                p_values['k1']['Hill'] = 0.3
                p_values['k1']['Mountain'] = 0.1
                print("Warning: Sum of morphological features is zero. Using default values.")

        # Verify column names in the soil data
        soil_columns = self.soil_data.columns
        if 'ROCK' not in soil_columns or 'NON_ROCK' not in soil_columns:
            print(f"Warning: Expected columns not found in soil data. Found: {soil_columns}")
            print("Using default soil proportions instead.")
            p_values['k2']['Rock'] = 0.3
            p_values['k2']['Non-Rock'] = 0.7
        else:
            # Calculate p values for soil type (k2)
            soil_sum = soil_data['ROCK'].sum() + soil_data['NON_ROCK'].sum()

            if soil_sum > 0:
                p_values['k2']['Rock'] = soil_data['ROCK'].sum() / soil_sum
                p_values['k2']['Non-Rock'] = soil_data['NON_ROCK'].sum() / soil_sum
            else:
                # Default values if sum is zero
                p_values['k2']['Rock'] = 0.3
                p_values['k2']['Non-Rock'] = 0.7
                print("Warning: Sum of soil types is zero. Using default values.")

        # Verify column names in the anthropisation data
        anthro_columns = self.anthropisation_data.columns
        non_anthro_col = 'NON-ANTHROPISED'
        anthro_col = 'ANTHROPISED'

        # Check for alternative spellings or formatting
        if non_anthro_col not in anthro_columns:
            possible_matches = [col for col in anthro_columns if 'non' in col.lower() and 'anthrop' in col.lower()]
            if possible_matches:
                non_anthro_col = possible_matches[0]
                print(f"Using '{non_anthro_col}' for non-anthropised data")
            else:
                print(f"Warning: Non-anthropised column not found. Available columns: {anthro_columns}")
                print("Using default anthropisation proportions instead.")
                p_values['k3']['Non-Anthropised'] = 0.5
                p_values['k3']['Anthropised'] = 0.5
                return p_values

        if anthro_col not in anthro_columns:
            possible_matches = [col for col in anthro_columns if 'anthrop' in col.lower() and 'non' not in col.lower()]
            if possible_matches:
                anthro_col = possible_matches[0]
                print(f"Using '{anthro_col}' for anthropised data")
            else:
                print(f"Warning: Anthropised column not found. Available columns: {anthro_columns}")
                print("Using default anthropisation proportions instead.")
                p_values['k3']['Non-Anthropised'] = 0.5
                p_values['k3']['Anthropised'] = 0.5
                return p_values

        # Calculate p values for anthropisation (k3)
        anthro_sum = anthro_data[non_anthro_col].sum() + anthro_data[anthro_col].sum()

        if anthro_sum > 0:
            p_values['k3']['Non-Anthropised'] = anthro_data[non_anthro_col].sum() / anthro_sum
            p_values['k3']['Anthropised'] = anthro_data[anthro_col].sum() / anthro_sum
        else:
            # Default values if sum is zero
            p_values['k3']['Non-Anthropised'] = 0.5
            p_values['k3']['Anthropised'] = 0.5
            print("Warning: Sum of anthropisation values is zero. Using default values.")

        return p_values

    def calculate_k_factor(self, timeframe, segment_id=None):
        """
        Calculate the k factor for a specific timeframe.

        The formula is: k_t = sum_{i,j} (p_{i,j} * q_{i,j,t})
        where p_{i,j} are calculated from geographical data
        and q_{i,j,t} are the cost coefficients from Table 1b.

        Args:
            timeframe (str): One of 'short-term', 'medium-term', or 'long-term'.
            segment_id (int, optional): ID of the pipeline segment to calculate k factor for.
                If None, calculates using average p values across all segments.

        Returns:
            float: The k factor for the specified timeframe.
        """
        if timeframe not in self.cost_coefficients:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of: {list(self.cost_coefficients.keys())}")

        # Calculate p values
        p_values = self.calculate_p_values(segment_id)

        if p_values is None:
            print(f"Warning: Could not calculate p values for segment {segment_id}. Using default k factor of 0.1.")
            return 0.1

        # Get q values (cost coefficients) for the specified timeframe
        q_values = self.cost_coefficients[timeframe]

        # Calculate k factor using the formula k_t = sum_{i,j} (p_{i,j} * q_{i,j,t})
        k_factor = 0.0

        # Sum over all geographical factor categories (k1, k2, k3)
        for factor_category, p_dict in p_values.items():
            q_dict = q_values[factor_category]

            # Sum over all classes within each category
            for factor_class, p_value in p_dict.items():
                q_value = q_dict[factor_class]
                k_factor += p_value * q_value

        return k_factor

    def save_cost_factors_to_csv(self, output_path='cost_factors.csv', max_segments=None):
        """
        Calculate and save cost factors for all timeframes to a CSV file.

        Args:
            output_path (str): Path to save the CSV file.
            max_segments (int, optional): Maximum number of segments to process.
                If None, processes all segments. Useful for limiting processing with large datasets.

        Returns:
            pandas.DataFrame: DataFrame containing the cost factors.
        """
        # Calculate cost factors for all segments and timeframes
        data = []

        # Get unique segment IDs from all datasets
        morph_segments = set(self.morphological_data['OID'].unique())
        soil_segments = set(self.soil_data['OID'].unique())
        anthro_segments = set(self.anthropisation_data['OID'].unique())

        # Find segments that are common to all datasets
        segment_ids = morph_segments.intersection(soil_segments).intersection(anthro_segments)

        print(f"Found {len(segment_ids)} segments common to all datasets")

        # Limit the number of segments if requested
        if max_segments is not None and max_segments < len(segment_ids):
            segment_ids = list(segment_ids)[:max_segments]
            print(f"Processing {max_segments} segments as requested")

        # Calculate cost factors for each segment
        for i, segment_id in enumerate(segment_ids):
            if i % 100 == 0:
                print(f"Processing segment {i} of {len(segment_ids)}")

            row = {'segment_id': segment_id}

            for timeframe in self.cost_coefficients.keys():
                k_factor = self.calculate_k_factor(timeframe, segment_id)
                row[f'k_factor_{timeframe}'] = k_factor

            data.append(row)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        print(f"Cost factors saved to {output_path}")
        return df

    def plot_k_factors_distribution(self, output_path='k_factors_distribution.png', max_segments=100):
        """
        Plot the distribution of k factors for all timeframes.

        Args:
            output_path (str): Path to save the plot.
            max_segments (int, optional): Maximum number of segments to process.

        Returns:
            None
        """
        # Calculate k factors for a subset of segments
        df = self.save_cost_factors_to_csv('temp_k_factors.csv', max_segments=max_segments)

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Plot histograms for each timeframe
        timeframes = ['short-term', 'medium-term', 'long-term']
        colors = ['#ff9999', '#66b3ff', '#99ff99']

        for timeframe, color in zip(timeframes, colors):
            col = f'k_factor_{timeframe}'
            plt.hist(df[col].dropna(), bins=30, alpha=0.7, label=timeframe, color=color)

        plt.xlabel('K Factor Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Geographical Cost Factors by Timeframe')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")

        # Print summary statistics
        print("\nSummary statistics for k factors:")
        for timeframe in timeframes:
            col = f'k_factor_{timeframe}'
            print(f"\n{timeframe}:")
            print(f"  Mean: {df[col].mean():.4f}")
            print(f"  Min: {df[col].min():.4f}")
            print(f"  Max: {df[col].max():.4f}")
            print(f"  Std: {df[col].std():.4f}")


class CO2PipelineMAS:
    """
    Class to calculate CO2 pipeline costs with geographical cost factors.
    Extends the CO2Chain_Oeuvray class with additional geographical factors.
    """

    def __init__(self, data_dir=None):
        """
        Initialize the CO2PipelineMAS class.

        Args:
            data_dir (str, optional): Directory containing the geographical data files.
                If None, tries to find the files in common locations.
        """
        # Initialize the geographical cost factors
        self.geo_factors = GeographicalCostFactors(data_dir)

        # Initialize the base CO2 pipeline cost calculator
        self.oeuvray_calculator = CO2Chain_Oeuvray()

    def calculate_cost(self, options, segment_id=None):
        """
        Calculate CO2 pipeline costs including geographical factors.

        Formula: co2_pipelines_mas = (1+k) * co2_pipelines_oeuvray

        Args:
            options (dict): Dictionary containing pipeline options:
                - timeframe (str): 'short-term', 'medium-term', or 'long-term'
                - length_km (float): Length of the pipeline in km
                - massflow_kg_per_s (float): Mass flow rate of CO2 in kg/s
                - terrain (str): 'Onshore' or 'Offshore'
                - electricity_price_eur_per_mw (float): Electricity price in EUR/MWh
                - operating_hours_per_a (int): Operating hours per year
                - p_inlet_bar (float): Inlet pressure in bar
                - p_outlet_bar (float): Outlet pressure in bar
                - discount_rate (float): Discount rate
            segment_id (int, optional): ID of the pipeline segment to use for geographical factors.
                If None, uses average geographical factors.

        Returns:
            dict: Dictionary containing cost results with geographical factors included.
        """
        # First, calculate the base cost using the Oeuvray model
        base_results = self.oeuvray_calculator.calculate_cost(options)

        # Calculate the geographical cost factor
        timeframe = options.get('timeframe', 'medium-term')
        k_factor = self.geo_factors.calculate_k_factor(timeframe, segment_id)

        # Apply the geographical cost factor
        # co2_pipelines_mas = (1+k) * co2_pipelines_oeuvray
        mas_results = {}

        # Apply the factor to the relevant cost components
        for key, value in base_results.items():
            if key == 'cost_pipeline':
                # Apply the factor to pipeline costs
                mas_results[key] = {
                    'unit_capex': value['unit_capex'] * (1 + k_factor),
                    'opex_var': value['opex_var'],
                    'opex_fix_abs': value['opex_fix_abs'] * (1 + k_factor),
                    'opex_fix_fraction': value['opex_fix_fraction'],
                    'lifetime': value['lifetime']
                }
            elif key == 'levelized_cost':
                # Apply the factor to the levelized cost
                mas_results[key] = value * (1 + k_factor)
            else:
                # Keep other values unchanged
                mas_results[key] = value

        # Add the k factor to the results
        mas_results['geographical_factor'] = k_factor

        return mas_results


# Example usage
if __name__ == "__main__":
    print("CO2 Pipeline Cost Model with Geographical Factors")
    print("=" * 50)

    # Create an instance of the GeographicalCostFactors class
    geo_factors = GeographicalCostFactors()

    # Generate a plot of the k factor distribution
    geo_factors.plot_k_factors_distribution(max_segments=50)

    # Create an instance of the CO2PipelineMAS class
    pipeline_calculator = CO2PipelineMAS()

    # Example pipeline options
    pipeline_options = {
        'timeframe': 'medium-term',
        'length_km': 100,
        'massflow_kg_per_s': 20,
        'terrain': 'Onshore',
        'electricity_price_eur_per_mw': 50,
        'operating_hours_per_a': 8000,
        'p_inlet_bar': 15,
        'p_outlet_bar': 80,
        'discount_rate': 0.08
    }

    # Get available segment IDs
    morph_segments = set(geo_factors.morphological_data['OID'].unique())
    soil_segments = set(geo_factors.soil_data['OID'].unique())
    anthro_segments = set(geo_factors.anthropisation_data['OID'].unique())
    common_segments = morph_segments.intersection(soil_segments).intersection(anthro_segments)

    if common_segments:
        # Select a segment ID that exists in all datasets
        segment_id = list(common_segments)[0]
        print(f"\nCalculating costs for segment {segment_id}...")

        # Calculate costs for a specific segment
        results = pipeline_calculator.calculate_cost(pipeline_options, segment_id)

        # Print results
        print(f"\nResults:")
        print(f"Geographical factor (k): {results['geographical_factor']:.4f}")
        print(f"Levelized cost: {results['levelized_cost']:.4f} EUR/t")
        print(f"Pipeline CAPEX: {results['cost_pipeline']['unit_capex']:.2f} EUR")
    else:
        print("\nNo common segments found across all datasets. Please check your data.")