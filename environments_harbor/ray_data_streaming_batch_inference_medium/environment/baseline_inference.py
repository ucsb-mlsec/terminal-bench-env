#!/usr/bin/env python3

import pandas as pd
import time
import csv

def transform_measurement(value):
    """
    Apply transformation to a single measurement.
    Transformation: multiply by 2.5 and add 10
    """
    return value * 2.5 + 10

def process_sensor_row(row):
    """
    Process a single sensor row sequentially.
    Takes the 9 numerical measurements, applies transformation to each,
    and returns the sum of all transformed values.
    """
    # Extract the 9 measurement columns (columns 1-9, assuming column 0 is ID)
    measurements = row[1:]
    
    # Transform each measurement one at a time (inefficient!)
    transformed_values = []
    for measurement in measurements:
        transformed = transform_measurement(float(measurement))
        transformed_values.append(transformed)
    
    # Sum all transformed values
    total = sum(transformed_values)
    
    # Round to 1 decimal place
    return round(total, 1)

def main():
    print("Starting baseline inference processing...")
    start_time = time.time()
    
    # Read the input CSV file
    print("Reading sensor data...")
    df = pd.read_csv('/workspace/data/sensor_readings.csv')
    
    total_rows = len(df)
    print(f"Total sensors to process: {total_rows}")
    
    # Prepare output list
    results = []
    
    # Process each row sequentially (very inefficient!)
    for idx, row in df.iterrows():
        # Print progress every 500 rows to show sequential processing
        if (idx + 1) % 500 == 0:
            print(f"Processing sensor {idx + 1} of {total_rows}...")
        
        # Get sensor ID (first column)
        sensor_id = row.iloc[0]
        
        # Get measurements (remaining columns)
        measurements = row.iloc[1:].values
        
        # Process this single row
        result = process_sensor_row(measurements)
        
        # Store result
        results.append({'id': sensor_id, 'result': result})
    
    print("Writing results to output file...")
    
    # Write results to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv('/workspace/baseline_output.csv', index=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Processing complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Processed {total_rows} sensors")
    print(f"Average time per sensor: {(elapsed_time/total_rows)*1000:.2f} milliseconds")
    print(f"Output written to: /workspace/baseline_output.csv")

if __name__ == "__main__":
    main()