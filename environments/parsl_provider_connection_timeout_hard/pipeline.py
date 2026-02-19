#!/usr/bin/env python3

import parsl
from parsl import python_app
import pandas as pd
import os
import sys
from pathlib import Path

# Import configuration from parsl_config.py
from parsl_config import config

# Load the Parsl configuration
parsl.load(config)

@python_app
def process_sensor_file(filename):
    """
    Process a sensor data CSV file and generate summary statistics.
    
    Args:
        filename: Path to the CSV file to process
        
    Returns:
        Path to the output file created
    """
    import pandas as pd
    from pathlib import Path
    
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Calculate summary statistics
    mean_temp = df['temperature'].mean()
    mean_pressure = df['pressure'].mean()
    count = len(df)
    
    # Create output directory if it doesn't exist
    output_dir = Path('/workspace/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    input_path = Path(filename)
    output_filename = output_dir / f"{input_path.stem}.txt"
    
    # Write results to output file
    with open(output_filename, 'w') as f:
        f.write(f"Sensor Data Summary for {input_path.name}\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Total Readings: {count}\n")
        f.write(f"Mean Temperature: {mean_temp:.2f}\n")
        f.write(f"Mean Pressure: {mean_pressure:.2f}\n")
    
    return str(output_filename)


def main():
    """Main execution function."""
    # Create output directory if it doesn't exist
    output_dir = Path('/workspace/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data directory
    data_dir = Path('/workspace/data')
    
    # List of sensor files to process
    sensor_files = [f'sensor_{i:03d}.csv' for i in range(1, 6)]
    
    print("Starting parallel sensor data processing...")
    print(f"Processing {len(sensor_files)} files...")
    
    # Submit all tasks for parallel processing
    futures = []
    for sensor_file in sensor_files:
        filepath = data_dir / sensor_file
        if filepath.exists():
            print(f"Submitting task for: {sensor_file}")
            future = process_sensor_file(str(filepath))
            futures.append(future)
        else:
            print(f"Warning: {sensor_file} not found", file=sys.stderr)
    
    # Wait for all tasks to complete
    print("\nWaiting for all tasks to complete...")
    results = []
    for i, future in enumerate(futures):
        try:
            result = future.result()
            results.append(result)
            print(f"Completed: {sensor_files[i]} -> {result}")
        except Exception as e:
            print(f"Error processing {sensor_files[i]}: {e}", file=sys.stderr)
    
    print(f"\nProcessing complete! {len(results)} files processed successfully.")


if __name__ == '__main__':
    main()