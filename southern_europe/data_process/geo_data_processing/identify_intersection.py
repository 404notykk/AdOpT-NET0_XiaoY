import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd
import numpy as np
from pathlib import Path
from defined_functions import (
    analyze_route_grid_intersections,
    export_to_excel,
    print_summary_statistics
)

#----- Load data -----#

# Load the shapefiles into GeoDataFrames
path_data_case_study = Path("../../northern_italy_data")

path_files_gis = path_data_case_study / "raw_data/gis_data"
path_files_node_flux = path_data_case_study / "geographical_feature"

route = gpd.read_file(path_files_gis / "routes_distances_pipeline.shp")
fishnet = gpd.read_file(path_files_gis / "fishnet_italy_25km.shp")

# Load node data for creating better sheet names
print("\n" + "="*50)
print("LOADING NODE DATA...")
print("="*50)

try:
    print(f"Looking for node file at: {path_files_node_flux / 'node_metrics.xlsx'}")
    network_nodes = pd.read_excel(path_files_node_flux/"node_metrics.xlsx", index_col=0, sheet_name='nodes') # nodes
    print("✓ Node data loaded successfully!")
    print(f"✓ Nodes: {network_nodes.index.nunique()} nodes")
    print("✓ Sample node data:")
    print(network_nodes[['node_name']].head())
    print(f"✓ Node data columns: {list(network_nodes.columns)}")
    print(f"✓ Index (node_id) sample: {network_nodes.index[:5].tolist()}")
except FileNotFoundError as e:
    print(f"❌ FileNotFoundError: {e}")
    print("❌ Will use original route names for sheets.")
    network_nodes = None
except Exception as e:
    print(f"❌ Error loading node data: {e}")
    print("❌ Will use original route names for sheets.")
    network_nodes = None

print("="*50)

# Add ID column to fishnet since it doesn't have one
fishnet['grid_id'] = fishnet.index

print("Geographic data loaded successfully!")
print(f"Routes: {len(route)} features")
print(f"Fishnet: {len(fishnet)} grids")

#----- Intersection analysis -----#

# Run intersection analysis
intersection_results = analyze_route_grid_intersections(route, fishnet)

# Export results - FIXED: Now passing network_nodes parameter!
output_file = path_data_case_study / "geographical_feature/route_grid_intersections.xlsx"
export_to_excel(intersection_results, output_file, network_nodes)

#----- Summary statistics -----#

print_summary_statistics(intersection_results, fishnet)

print(f"\n{'=' * 60}")
print("ANALYSIS COMPLETE!")
print(f"Results saved to: {output_file}")
print(f"{'=' * 60}")