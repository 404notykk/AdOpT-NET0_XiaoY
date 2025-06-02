import rasterio
from rasterio.enums import Resampling
import numpy as np
import os

# Input and output paths
input_tif = "/Users/ykk/Desktop/Thesis/data/extracted_elevation_europe.tif"
output_tif = "/Users/ykk/Desktop/Thesis/data/extracted_elevation_europe_1km.tif"


def resample_raster(input_path, output_path, target_resolution=1000):
    """
    Resample a raster to a target resolution in meters.
    For geographic CRS (EPSG:4258), we'll approximate 1km as 0.01 degrees
    """
    with rasterio.open(input_path) as src:
        # Get the source CRS
        src_crs = src.crs

        # Print original info
        print(f"Original dimensions: {src.width}x{src.height}")
        print(f"Original resolution: {src.res}")

        # For geographic CRS, target_resolution should be in degrees
        # Approximate 1km as 0.009 degrees (rough approximation)
        if src_crs.is_geographic:
            target_res_degrees = 0.009
            print(f"Using target resolution: {target_res_degrees} degrees (approx. 1km)")
        else:
            # For projected CRS, use meters directly
            target_res_degrees = target_resolution
            print(f"Using target resolution: {target_res_degrees} meters")

        # CORRECTED: Calculate the scaling factor - this was inverted before
        scale_factor_x = target_res_degrees / src.res[0]
        scale_factor_y = target_res_degrees / src.res[1]

        # Calculate new dimensions
        new_width = int(src.width / scale_factor_x)
        new_height = int(src.height / scale_factor_y)

        print(f"New dimensions: {new_width}x{new_height}")

        # Define the output transform
        new_transform = rasterio.Affine(
            target_res_degrees, src.transform.b, src.transform.c,
            src.transform.d, -target_res_degrees, src.transform.f
        )

        # Create the output metadata
        out_meta = src.meta.copy()
        out_meta.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
            'compress': 'lzw',
            'predictor': 3,
            'tiled': True
        })

        # Perform the resampling
        print("Resampling the data...")
        resampled_data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.bilinear
        )

        # Save the resampled raster
        print(f"Saving resampled data to {output_path}")
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(resampled_data, 1)

        # Calculate size reduction
        original_size = os.path.getsize(input_path) / (1024 ** 3)  # GB
        new_size = os.path.getsize(output_path) / (1024 ** 3)  # GB
        reduction = (1 - new_size / original_size) * 100

        print(f"Original size: {original_size:.2f} GB")
        print(f"New size: {new_size:.2f} GB")
        print(f"Size reduction: {reduction:.2f}%")

        return resampled_data


try:
    print("Starting resampling process...")
    resampled = resample_raster(input_tif, output_tif)

    # Calculate some statistics on the resampled data
    valid_data = resampled[~np.isnan(resampled)]
    if len(valid_data) > 0:
        print(f"Resampled data statistics:")
        print(f"  Min: {np.min(valid_data)}")
        print(f"  Max: {np.max(valid_data)}")
        print(f"  Mean: {np.mean(valid_data)}")
        print(f"  Median: {np.median(valid_data)}")

    print("Resampling completed successfully!")

except Exception as e:
    print(f"Error during resampling: {e}")