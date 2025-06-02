import rasterio
from rasterio.enums import Resampling


def resample_raster(input_path, output_path, scale_factor=0.025):
    """
    Resample a raster by a given scale factor.
    For 25m to 1000m (1km), scale_factor = 0.025
    """
    with rasterio.open(input_path) as dataset:
        # Get metadata
        profile = dataset.profile

        # Calculate new dimensions
        new_height = int(dataset.height * scale_factor)
        new_width = int(dataset.width * scale_factor)

        # Update profile for the output file
        profile.update(
            height=new_height,
            width=new_width,
            transform=dataset.transform * dataset.transform.scale(
                (dataset.width / new_width),
                (dataset.height / new_height)
            )
        )

        # Resample data
        data = dataset.read(
            out_shape=(dataset.count, new_height, new_width),
            resampling=Resampling.bilinear
        )

        # Write the resampled data to the output file
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)

        print(f"Successfully resampled {input_path} to {output_path}")
        print(f"Original size: {dataset.height} x {dataset.width}")
        print(f"Resampled size: {new_height} x {new_width}")


# Input and output paths
input_raster = "/Users/ykk/Desktop/Thesis/data/EUD_CP_SLOP_mosaic/eudem_slop_3035_europe.tif"
output_raster = "/Users/ykk/Desktop/Thesis/data/eudem_slop_3035_europe_1km.tif"

# Run the resampling (25m to 1km = factor of 0.025)
resample_raster(input_raster, output_raster, scale_factor=0.025)