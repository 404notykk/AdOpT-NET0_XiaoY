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
# Configure the sizes & file names here (adjust if your filenames differ)
# For 25 km, if your files have suffixes, update those paths accordingly.
# ----------------------------
SIZES = {
    5: {
        "fishnet": path_files_gis / "fishnet_greece_5km.shp",
        "anthro":  path_files_grids / "anthropisation_grids_greece_5km.csv",
        "soil":    path_files_grids / "soil_type_grids_greece_5km.csv",
        "morpho":  path_files_grids / "morphological_feature_grids_greece_5km.csv",
    },
    10: {
        "fishnet": path_files_gis / "fishnet_greece_10km.shp",
        "anthro":  path_files_grids / "anthropisation_grids_greece_10km.csv",
        "soil":    path_files_grids / "soil_type_grids_greece_10km.csv",
        "morpho":  path_files_grids / "morphological_feature_grids_greece_10km.csv",
    },
    25: {
        "fishnet": path_files_gis / "fishnet_greece_25km.shp",
        "anthro":  path_files_grids / "anthropisation_grids_greece.csv",
        "soil":    path_files_grids / "soil_type_grids_greece.csv",
        "morpho":  path_files_grids / "morphological_feature_grids_greece.csv",
    },
    50: {
        "fishnet": path_files_gis / "fishnet_greece_50km.shp",
        "anthro":  path_files_grids / "anthropisation_grids_greece_50km.csv",
        "soil":    path_files_grids / "soil_type_grids_greece_50km.csv",
        "morpho":  path_files_grids / "morphological_feature_grids_greece_50km.csv",
    },
}

# ----------------------------
# Helpers
# ----------------------------
def ensure_grid_id(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize ID column to GRID_OID for merging."""
    if "GRID_OID" in df.columns:
        return df
    if "GRID_ID" in df.columns:
        return df.rename(columns={"GRID_ID": "GRID_OID"})
    raise KeyError("Neither 'GRID_OID' nor 'GRID_ID' found in dataframe.")

def load_and_compute(fishnet_path: Path,
                     anthro_csv: Path,
                     soil_csv: Path,
                     morpho_csv: Path,
                     boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Inner-merge fishnet with CSVs, compute factors, clip to boundary."""
    fishnet = gpd.read_file(fishnet_path)
    if "GRID_OID" not in fishnet.columns and "GRID_ID" not in fishnet.columns:
        fishnet = fishnet.reset_index().rename(columns={"index": "GRID_OID"})
    fishnet = ensure_grid_id(fishnet)

    anthro = ensure_grid_id(pd.read_csv(anthro_csv))
    soil   = ensure_grid_id(pd.read_csv(soil_csv))
    morpho = ensure_grid_id(pd.read_csv(morpho_csv))

    # Inner joins — safe because you've cleaned CSVs to shared IDs
    g = (fishnet
         .merge(anthro, on="GRID_OID", how="inner")
         .merge(soil,   on="GRID_OID", how="inner", suffixes=("", "_soil"))
         .merge(morpho, on="GRID_OID", how="inner", suffixes=("", "_morpho")))

    # Compute factors (your weights)
    # Anthro
    req_a = {"NON_ANTHROPISED_A", "ANTHROPISED_A"}
    missing = req_a - set(g.columns)
    if missing:
        raise KeyError(f"Anthro columns missing: {missing}")
    g["ANTHRO_FACTOR"] = 0.0025 * g["NON_ANTHROPISED_A"] + 0.38 * g["ANTHROPISED_A"]

    # Soil
    req_s = {"NON_ROCK_S", "ROCK_S"}
    missing = req_s - set(g.columns)
    if missing:
        raise KeyError(f"Soil columns missing: {missing}")
    g["SOIL_FACTOR"] = 0.025 * g["NON_ROCK_S"] + 0.21 * g["ROCK_S"]

    # Morphological
    req_m = {"PLAIN_M", "HILL_M", "MOUNTAIN_M"}
    missing = req_m - set(g.columns)
    if missing:
        raise KeyError(f"Morpho columns missing: {missing}")
    g["MORPH_FACTOR"] = 0.025 * g["PLAIN_M"] + 0.06 * g["HILL_M"] + 0.09 * g["MOUNTAIN_M"]

    # Integrated cost factor (sum of three)
    g["COST_FACTOR"] = g["ANTHRO_FACTOR"] + g["SOIL_FACTOR"] + g["MORPH_FACTOR"]

    # CRS harmonization & clip
    if g.crs != boundary_gdf.crs:
        g = g.to_crs(boundary_gdf.crs)
    g = gpd.clip(g, boundary_gdf)
    return g

# ----------------------------
# Load & compute for all sizes
# ----------------------------
layers_by_size = {}
for size, p in SIZES.items():
    layers_by_size[size] = load_and_compute(
        fishnet_path=p["fishnet"],
        anthro_csv=p["anthro"],
        soil_csv=p["soil"],
        morpho_csv=p["morpho"],
        boundary_gdf=greece
    )

# ----------------------------
# Figure-maker: one factor per figure, four subplots for 4 sizes
# ----------------------------
def plot_factor_across_sizes(layers_by_size, factor_col: str, title_prefix: str, outfile: str):
    # Build a per-factor color scale across all sizes
    vals = pd.concat([gdf[factor_col] for gdf in layers_by_size.values()], ignore_index=True)
    norm = Normalize(vmin=vals.min(), vmax=vals.max())
    cmap = cmc.navia_r

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=False)
    plt.subplots_adjust(wspace=0.03, right=0.9)

    order = [5, 10, 25, 50]
    for ax, size in zip(axes, order):
        gdf = layers_by_size[size]
        greece.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=1)
        gdf.plot(column=factor_col, ax=ax, cmap=cmap, norm=norm, legend=False, zorder=2)
        gdf.boundary.plot(ax=ax, color="gray", linewidth=0.2, alpha=0.5, zorder=3)
        ax.set_title(f"{title_prefix} — {size} km", fontsize=12)
        ax.set_axis_off()

    # Shared colorbar
    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Factor Value", fontsize=12)

    fig.suptitle(f"{title_prefix} by fishnet size", fontsize=14, y=0.98)
    plt.savefig(outfile, dpi=500, bbox_inches="tight")
    plt.show()
    print(f"[OK] Saved {outfile}")

# ----------------------------
# Generate the four visualizations (one per factor)
# ----------------------------
plot_factor_across_sizes(layers_by_size, "ANTHRO_FACTOR", "Anthropisation factor", "greece_anthro_by_cellsize.png")
plot_factor_across_sizes(layers_by_size, "SOIL_FACTOR",   "Soil factor",           "greece_soil_by_cellsize.png")
plot_factor_across_sizes(layers_by_size, "MORPH_FACTOR",  "Geomorphology factor",  "greece_morpho_by_cellsize.png")
plot_factor_across_sizes(layers_by_size, "COST_FACTOR",   "Integrated cost factor","greece_cost_by_cellsize.png")
