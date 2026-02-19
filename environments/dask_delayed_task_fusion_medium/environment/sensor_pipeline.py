#!/usr/bin/env python3
"""
Inefficient sensor data pipeline - creates excessive tasks for demonstration
This version intentionally fragments the task graph by creating separate tasks
for each transformation step.
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import dask
from dask import delayed


# Inefficient version - each operation is a separate task
def read_sensor_file(filepath):
    """Read a sensor CSV file"""
    return pd.read_csv(filepath)


def extract_temperature(df):
    """Extract temperature column"""
    return df['temperature']


def extract_humidity(df):
    """Extract humidity column"""
    return df['humidity']


def extract_pressure(df):
    """Extract pressure column"""
    return df['pressure']


def convert_temp_to_celsius(temp_series):
    """Convert temperature from Fahrenheit to Celsius"""
    return (temp_series - 32) * 5/9


def normalize_humidity(humidity_series):
    """Normalize humidity to 0-1 range"""
    min_val = humidity_series.min()
    max_val = humidity_series.max()
    if max_val - min_val == 0:
        return humidity_series * 0
    return (humidity_series - min_val) / (max_val - min_val)


def normalize_pressure(pressure_series):
    """Normalize pressure to 0-1 range"""
    min_val = pressure_series.min()
    max_val = pressure_series.max()
    if max_val - min_val == 0:
        return pressure_series * 0
    return (pressure_series - min_val) / (max_val - min_val)


def calculate_temp_stats(temp_series):
    """Calculate statistics for temperature"""
    return {
        'mean': float(temp_series.mean()),
        'std': float(temp_series.std()),
        'min': float(temp_series.min()),
        'max': float(temp_series.max())
    }


def calculate_humidity_stats(humidity_series):
    """Calculate statistics for humidity"""
    return {
        'mean': float(humidity_series.mean()),
        'std': float(humidity_series.std()),
        'min': float(humidity_series.min()),
        'max': float(humidity_series.max())
    }


def calculate_pressure_stats(pressure_series):
    """Calculate statistics for pressure"""
    return {
        'mean': float(pressure_series.mean()),
        'std': float(pressure_series.std()),
        'min': float(pressure_series.min()),
        'max': float(pressure_series.max())
    }


def merge_stats(temp_stats, humidity_stats, pressure_stats):
    """Merge all statistics into one dictionary"""
    return {
        'temperature': temp_stats,
        'humidity': humidity_stats,
        'pressure': pressure_stats
    }


def combine_all_results(results_list):
    """Aggregate statistics across all sensors"""
    if not results_list:
        return {}
    
    # Aggregate each metric type
    all_temp_means = [r['temperature']['mean'] for r in results_list]
    all_temp_stds = [r['temperature']['std'] for r in results_list]
    all_humidity_means = [r['humidity']['mean'] for r in results_list]
    all_humidity_stds = [r['humidity']['std'] for r in results_list]
    all_pressure_means = [r['pressure']['mean'] for r in results_list]
    all_pressure_stds = [r['pressure']['std'] for r in results_list]
    
    return {
        'num_sensors': len(results_list),
        'temperature': {
            'overall_mean': float(np.mean(all_temp_means)),
            'overall_std': float(np.mean(all_temp_stds)),
            'mean_of_means': float(np.mean(all_temp_means))
        },
        'humidity': {
            'overall_mean': float(np.mean(all_humidity_means)),
            'overall_std': float(np.mean(all_humidity_stds)),
            'mean_of_means': float(np.mean(all_humidity_means))
        },
        'pressure': {
            'overall_mean': float(np.mean(all_pressure_means)),
            'overall_std': float(np.mean(all_pressure_stds)),
            'mean_of_means': float(np.mean(all_pressure_means))
        }
    }


def process_all_sensors():
    """
    Main processing function - INEFFICIENT VERSION
    Creates 11 tasks per input file, resulting in excessive task graph fragmentation
    """
    # Find all sensor files
    sensor_files = sorted(glob.glob('/workspace/data/sensor_*.csv'))
    
    if not sensor_files:
        print("No sensor files found!")
        return {}
    
    print(f"Processing {len(sensor_files)} sensor files (inefficient version)...")
    
    # Process each file with excessive task fragmentation
    all_results = []
    
    for filepath in sensor_files:
        # Each step is a separate delayed task - INEFFICIENT!
        df = delayed(read_sensor_file)(filepath)
        
        # Extract each column separately - creates 3 tasks
        temp = delayed(extract_temperature)(df)
        humidity = delayed(extract_humidity)(df)
        pressure = delayed(extract_pressure)(df)
        
        # Transform each column separately - creates 3 more tasks
        temp_celsius = delayed(convert_temp_to_celsius)(temp)
        humidity_norm = delayed(normalize_humidity)(humidity)
        pressure_norm = delayed(normalize_pressure)(pressure)
        
        # Calculate stats separately - creates 3 more tasks
        temp_stats = delayed(calculate_temp_stats)(temp_celsius)
        humidity_stats = delayed(calculate_humidity_stats)(humidity_norm)
        pressure_stats = delayed(calculate_pressure_stats)(pressure_norm)
        
        # Merge stats - creates 1 more task
        merged = delayed(merge_stats)(temp_stats, humidity_stats, pressure_stats)
        
        all_results.append(merged)
    
    # Final aggregation - creates 1 task
    final_result = delayed(combine_all_results)(all_results)
    
    # Compute the entire task graph
    result = final_result.compute()
    
    return result


if __name__ == '__main__':
    results = process_all_sensors()
    print("\nFinal Results:")
    print(json.dumps(results, indent=2))