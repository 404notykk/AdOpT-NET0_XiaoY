import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import cmcrameri.cm as cmc
from pathlib import Path
from matplotlib.colors import Normalize

# ----------------------------
# Paths
# ----------------------------
path_data_case_study = Path("../greece_data")
path_files_gis = path_data_case_study / "raw_data/gis_data"
path_files_grids = path_data_case_study / "geographical_feature"

# Greece boundary (used for clipping and context)
greece = gpd.read_file(path_files_gis / "greece_WGS1984.shp")

# ----------------------------
# Configure the sizes & file names here
# Update the paths if your files have different names
# ----------------------------
SIZES = {
    5: {
        "fishnet": path_files_gis / "fishnet_greece_5km.shp",
        "anthro":  path_files_grids / "anthropisation_grids_greece_5km.csv",
    },
    10: {
        "fishnet": path_files_gis / "fishnet_greece_10km.shp",
        "anthro":  path_files_grids / "anthropisation_grids_greece_10km.csv",
    },
    25: {
        "fishnet": path_files_gis / "fishnet_greece_25km.shp",
        "anthro":  path_files_grids / "anthropisation_grids_greece.csv",
    },
    50: {
        "fishnet": path_files_gis / "fishnet_greece_50km.shp",
        "anthro":  path_files_grids / "anthropisation_grids_greece_50km.csv",
    },
}

# ----------------------------
# Helper to load, merge, compute ANTHRO_FACTOR, and clip
# ----------------------------
def load_anthro_layer(fishnet_path: Path, anthro_csv: Path) -> gpd.GeoDataFrame:
    # Load fishnet; ensure GRID_OID exists (use index if needed)
    fishnet = gpd.read_file(fishnet_path).reset_index().rename(columns={"index": "GRID_OID"})
    if "GRID_OID" not in fishnet.columns:
        raise ValueError(f"'GRID_OID' not found in {fishnet_path.name}. Ensure the grid has a unique ID column.")

    # Load anthro CSV and merge
    anthro = pd.read_csv(anthro_csv)

    # Check minimal columns expected from your original script
    required_cols = {"GRID_OID", "NON_ANTHROPISED_A", "ANTHROPISED_A"}
    missing = required_cols - set(anthro.columns)
    if missing:
        raise ValueError(f"Missing columns in {anthro_csv.name}: {missing}")

    layer = fishnet.merge(anthro, on="GRID_OID", how="left")

    # Compute only the anthropisation factor
    layer["ANTHRO_FACTOR"] = 0.0025 * layer["NON_ANTHROPISED_A"] + 0.38 * layer["ANTHROPISED_A"]

    # Clip to Greece boundary
    # Ensure CRS match
    if layer.crs != greece.crs:
        layer = layer.to_crs(greece.crs)
    clipped = gpd.clip(layer, greece)
    return clipped

# ----------------------------
# Load all layers by size
# ----------------------------
layers_by_size = {}
for size, paths in SIZES.items():
    layers_by_size[size] = load_anthro_layer(paths["fishnet"], paths["anthro"])

# ----------------------------
# Establish a shared color scale across all sizes
# ----------------------------
all_vals = pd.concat([gdf["ANTHRO_FACTOR"] for gdf in layers_by_size.values()], ignore_index=True)
vmin, vmax = all_vals.min(), all_vals.max()
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = cmc.navia_r

# ----------------------------
# Plot: 2x2 subplots for the four cell sizes (ANTHRO_FACTOR only)
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# Keep a stable order for sizes in the figure
plot_order = [5, 10, 25, 50]

for ax, size in zip(axes, plot_order):
    gdf = layers_by_size[size]
    # Draw national boundary for context
    greece.boundary.plot(ax=ax, color="black", linewidth=0.8)

    # Draw the fishnet colored by ANTHRO_FACTOR
    gdf.plot(column="ANTHRO_FACTOR", ax=ax, cmap=cmap, norm=norm, legend=False)
    gdf.boundary.plot(ax=ax, color="gray", linewidth=0.2, alpha=0.5)

    ax.set_title(f"Anthropisation factor — {size} km grid", fontsize=12)
    ax.set_axis_off()

# Shared colorbar
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Anthropisation Factor", fontsize=12)

plt.subplots_adjust(wspace=0.02, hspace=0.08, right=0.9)
plt.savefig("greece_anthro_factor_by_cellsize.png", dpi=600, bbox_inches="tight")
plt.show()
