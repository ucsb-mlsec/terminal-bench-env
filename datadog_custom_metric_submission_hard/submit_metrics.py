#!/usr/bin/env python3

import json
import os
import requests
import sys

# This script submits application metrics to Datadog API
# Should work fine after last week's updates

def read_metrics_file():
    """Read metrics from the input JSON file"""
    # Bug 1: Wrong file path - missing /workspace/ prefix
    metrics_file = 'metrics_data.json'
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        # Bug 2: Returns None instead of handling error properly
        return None

def format_metrics_for_datadog(raw_metrics):
    """Transform metrics into Datadog API format"""
    formatted = {'series': []}
    
    # Bug 3: Assuming raw_metrics is a list but not checking structure
    for metric in raw_metrics:
        # Bug 4: Wrong key name - using 'name' instead of 'metric'
        metric_entry = {
            'name': metric.get('metric_name'),
            # Bug 5: Points should be list of lists [[timestamp, value]] but creating wrong structure
            'points': [metric.get('timestamp'), metric.get('value')],
            'tags': metric.get('tags', [])
        }
        formatted['series'].append(metric_entry)
    
    return formatted

def submit_to_datadog(payload):
    """Submit metrics to Datadog API endpoint"""
    
    # Bug 6: Wrong environment variable name
    api_key = os.getenv('DATADOG_API_KEY')
    
    if not api_key:
        print("Warning: API key not found, using default")
        api_key = ""
    
    endpoint = 'https://api.datadoghq.com/api/v1/series'
    
    # Bug 7: Wrong header name - should be 'DD-API-KEY' not 'X-API-KEY'
    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': api_key
    }
    
    # Save payload for verification (Bug 8: saves incorrectly formatted data)
    try:
        with open('/workspace/fixed_output.json', 'w') as f:
            # Bug 9: Not using json.dumps, just converting to string
            f.write(str(payload))
    except Exception as e:
        print(f"Could not save output: {e}")
    
    print("Submitting metrics to Datadog...")
    print(f"Payload: {payload}")
    
    try:
        # Bug 10: Using GET instead of POST
        response = requests.get(endpoint, headers=headers, json=payload)
        
        if response.status_code == 202:
            print("Metrics submitted successfully!")
            return True
        else:
            print(f"Failed to submit metrics: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error submitting metrics: {e}")
        return False

def main():
    """Main execution function"""
    print("Starting metrics submission process...")
    
    # Read the metrics data
    raw_data = read_metrics_file()
    
    # Bug 11: Not checking if raw_data is None before proceeding
    metrics_list = raw_data.get('metrics', [])
    
    if not metrics_list:
        print("No metrics found to submit")
        sys.exit(1)
    
    print(f"Found {len(metrics_list)} metrics to submit")
    
    # Format metrics for Datadog API
    payload = format_metrics_for_datadog(metrics_list)
    
    # Submit to Datadog
    success = submit_to_datadog(payload)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()