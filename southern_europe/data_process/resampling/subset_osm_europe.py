import os
import subprocess
import psutil
import time
import sys


def extract_osm_region(input_file, output_file, north, west, south, east):
    """
    Extract a specific region from an OSM PBF file using osmium.

    Parameters:
    -----------
    input_file : str
        Path to the input OSM PBF file
    output_file : str
        Path to the output OSM PBF file
    north, west, south, east : float
        Coordinates of the bounding box
    """
    start_time = time.time()

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")

    # Check file size
    input_size_gb = os.path.getsize(input_file) / (1024 ** 3)
    print(f"Input file size: {input_size_gb:.2f} GB")

    # Check available disk space
    disk_usage = psutil.disk_usage(os.path.dirname(output_file))
    free_space_gb = disk_usage.free / (1024 ** 3)
    print(f"Free disk space: {free_space_gb:.2f} GB")

    if free_space_gb < input_size_gb * 1.5:
        print(f"WARNING: Low disk space. Recommended: {input_size_gb * 1.5:.2f} GB, Available: {free_space_gb:.2f} GB")
        choice = input("Continue anyway? (y/n): ")
        if choice.lower() != 'y':
            sys.exit("Extraction cancelled due to insufficient disk space")

    # Check available memory
    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"Available memory: {available_memory_gb:.2f} GB")

    if available_memory_gb < 8:
        print("WARNING: Less than 8GB of RAM available. Extraction might be slow or fail.")
        choice = input("Continue anyway? (y/n): ")
        if choice.lower() != 'y':
            sys.exit("Extraction cancelled due to insufficient memory")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build osmium command - removed the threads parameter
    cmd = [
        "osmium", "extract",
        "--strategy", "complete_ways",
        "--bbox", f"{west},{south},{east},{north}",
        "--progress",
        "--overwrite",
        "-o", output_file,
        input_file
    ]

    try:
        # Print command for reference
        print(f"Running command: {' '.join(cmd)}")
        print(f"Extraction started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("This may take several hours for a 32GB dataset...")

        # Run the command with progress output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            sys.stdout.flush()

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        # Calculate and report statistics
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\nExtraction completed successfully!")
        print(f"Elapsed time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        # Report output file size
        if os.path.exists(output_file):
            output_size_gb = os.path.getsize(output_file) / (1024 ** 3)
            print(f"Output file size: {output_size_gb:.2f} GB")
            print(f"Reduction ratio: {output_size_gb / input_size_gb:.2%}")
        else:
            print("WARNING: Output file not found!")

    except subprocess.CalledProcessError as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExtraction was interrupted by user.")
        print("The output file may be incomplete.")
        sys.exit(1)


if __name__ == "__main__":
    # Input file path
    input_file = "/Users/ykk/Desktop/Thesis/data/europe-latest.osm.pbf"

    # Output file path
    output_file = "/Users/ykk/Desktop/Thesis/data/mediterranean-extract.osm.pbf"

    # Bounding box coordinates for Mediterranean region
    north = 47.5
    west = 5.9
    south = 34.4
    east = 30.6

    # Report CPU count but don't use threads parameter
    cpu_count = psutil.cpu_count(logical=True)
    print(f"System has {cpu_count} logical CPUs.")

    # Extract the region
    extract_osm_region(
        input_file=input_file,
        output_file=output_file,
        north=north,
        west=west,
        south=south,
        east=east
    )

    print("Process complete!")
    print(f"Extracted region saved to: {output_file}")