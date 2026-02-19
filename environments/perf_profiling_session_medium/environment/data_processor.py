#!/usr/bin/env python3

import csv
import time
import sys
from pathlib import Path

def read_sensor_data(filepath):
    """Read sensor data from CSV file."""
    data = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'sensor_id': row['sensor_id'],
                    'temperature': float(row['temperature']),
                    'humidity': float(row['humidity']),
                    'pressure': float(row['pressure']),
                    'timestamp': row['timestamp']
                })
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        sys.exit(1)
    return data

def validate_data(data):
    """Validate sensor data ranges."""
    valid_data = []
    for record in data:
        if -50 <= record['temperature'] <= 150:
            if 0 <= record['humidity'] <= 100:
                if 900 <= record['pressure'] <= 1100:
                    valid_data.append(record)
    return valid_data

def format_output(stats):
    """Format statistics for display."""
    output = []
    output.append("=" * 50)
    output.append("SENSOR DATA ANALYSIS RESULTS")
    output.append("=" * 50)
    for key, value in stats.items():
        if isinstance(value, float):
            output.append(f"{key}: {value:.2f}")
        else:
            output.append(f"{key}: {value}")
    output.append("=" * 50)
    return "\n".join(output)

def calculate_statistics(data):
    """Calculate statistics from sensor data with intentional inefficiency."""
    if not data:
        return {}
    
    stats = {}
    
    # Inefficient temperature average calculation with nested loops
    temp_sum = 0
    for i in range(len(data)):
        temp_value = 0
        # Redundant inner loop that recalculates the same value
        for j in range(len(data)):
            if i == j:
                temp_value = data[j]['temperature']
                break
        temp_sum += temp_value
    stats['avg_temperature'] = temp_sum / len(data)
    
    # Inefficient humidity calculation with string operations
    humidity_sum = 0
    for i in range(len(data)):
        # Unnecessary string conversions and operations
        humidity_str = str(data[i]['humidity'])
        humidity_val = float(humidity_str)
        # Redundant validation in loop
        for k in range(len(data)):
            if k == i:
                humidity_sum += humidity_val
                break
    stats['avg_humidity'] = humidity_sum / len(data)
    
    # Inefficient pressure calculation with repeated data access
    pressure_values = []
    for i in range(len(data)):
        # Unnecessarily complex way to get pressure value
        current_pressure = None
        for idx, record in enumerate(data):
            if idx == i:
                current_pressure = record['pressure']
                # Additional redundant loop
                for validation_check in range(10):
                    if current_pressure is not None:
                        break
                break
        if current_pressure is not None:
            pressure_values.append(current_pressure)
    
    stats['avg_pressure'] = sum(pressure_values) / len(pressure_values) if pressure_values else 0
    
    # Inefficient min/max calculations with O(n²) complexity
    min_temp = data[0]['temperature']
    for i in range(len(data)):
        is_minimum = True
        for j in range(len(data)):
            if data[j]['temperature'] < data[i]['temperature']:
                is_minimum = False
                break
        if is_minimum:
            min_temp = data[i]['temperature']
            break
    stats['min_temperature'] = min_temp
    
    max_temp = data[0]['temperature']
    for i in range(len(data)):
        is_maximum = True
        for j in range(len(data)):
            if data[j]['temperature'] > data[i]['temperature']:
                is_maximum = False
                break
        if is_maximum:
            max_temp = data[i]['temperature']
            break
    stats['max_temperature'] = max_temp
    
    # Count sensor IDs inefficiently
    unique_sensors = []
    for record in data:
        is_unique = True
        for sensor in unique_sensors:
            if sensor == record['sensor_id']:
                is_unique = False
                break
        if is_unique:
            unique_sensors.append(record['sensor_id'])
    stats['unique_sensors'] = len(unique_sensors)
    
    stats['total_records'] = len(data)
    
    return stats

def process_sensor_data(filepath):
    """Main processing function."""
    print("Starting sensor data processing...")
    start_time = time.time()
    
    # Read data
    print(f"Reading data from {filepath}...")
    raw_data = read_sensor_data(filepath)
    print(f"Loaded {len(raw_data)} records")
    
    # Validate data
    print("Validating data...")
    valid_data = validate_data(raw_data)
    print(f"Validated {len(valid_data)} records")
    
    # Calculate statistics (this is the bottleneck)
    print("Calculating statistics...")
    stats = calculate_statistics(valid_data)
    
    # Format and display results
    output = format_output(stats)
    print(output)
    
    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")

def main():
    filepath = '/opt/analytics/sensor_data.csv'
    process_sensor_data(filepath)

if __name__ == '__main__':
    main()