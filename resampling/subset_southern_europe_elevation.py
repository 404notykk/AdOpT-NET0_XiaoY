import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
import os



# Define input and output file paths
input_tif = "/Users/ykk/Desktop/Thesis/data/EU_DEM_mosaic_5deg/eudem_dem_4258_europe.tif"
output_tif = "/Users/ykk/Desktop/Thesis/data/extracted_elevation_europe.tif"

# # Check data file
# with rasterio.open(input_tif) as src:
#     print(f"CRS: {src.crs}")
#     print(f"Transform: {src.transform}")
#     print(f"Resolution: {src.res}")
#     print(f"Dimensions: {src.width}x{src.height}")
#     print(f"Number of bands: {src.count}")
#     print(f"Data type: {src.dtypes[0]}")
#     print(f"NoData value: {src.nodata}")

# Define input and output file paths
input_tif = "/Users/ykk/Desktop/Thesis/data/EU_DEM_mosaic_5deg/eudem_dem_4258_europe.tif"
output_tif = "/Users/ykk/Desktop/Thesis/data/extracted_elevation_europe.tif"

# Define the boundary coordinates (west, south, east, north)
bounds = (5.9, 34.4, 30.6, 47.5)


def extract_by_bounds_windowed(tif_path, bounds, output_path):
    with rasterio.open(tif_path) as src:
        # Convert geographic bounds to pixel coordinates
        window = from_bounds(*bounds, src.transform)

        # Ensure window boundaries are integers
        window = window.round_offsets().round_lengths()

        print(f"Reading window: {window}")

        # Read the data using the window
        data = src.read(1, window=window)

        print(f"Extracted data shape: {data.shape}")

        # Calculate the new transform for the windowed data
        transform = rasterio.windows.transform(window, src.transform)

        # Create metadata for the output
        out_meta = src.meta.copy()
        out_meta.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform
        })

        # Save the output
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(data, 1)

        print(f"Saved extracted data to {output_path}")

    return data, out_meta


# Execute the extraction
try:
    print("Starting extraction using windowed approach...")
    data, metadata = extract_by_bounds_windowed(input_tif, bounds, output_tif)

    # Calculate some statistics ignoring NaN values
    valid_data = data[~np.isnan(data)]
    if len(valid_data) > 0:
        print(f"Data statistics:")
        print(f"  Min: {np.min(valid_data)}")
        print(f"  Max: {np.max(valid_data)}")
        print(f"  Mean: {np.mean(valid_data)}")
        print(f"  Median: {np.median(valid_data)}")
    else:
        print("Warning: No valid data found in extraction area!")

    print("Extraction completed successfully!")

except Exception as e:
    print(f"Error during extraction: {e}")