#!/usr/bin/env python3

import requests
import json
import time
import sys
import logging
import psutil

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Datadog API configuration
DATADOG_API_KEY = 'dd_api_key_12345'
DATADOG_APP_KEY = 'dd_app_key_67890'
DATADOG_API_URL = 'https://api.datadoghq.com/api/v1/series'

def collect_metrics():
    """Collect system metrics and return as dictionary."""
    current_time = int(time.time())
    
    metrics = {
        'cpu_usage': psutil.cpu_percent(interval=1),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'timestamp': current_time
    }
    
    logging.info(f"Collected metrics: {metrics}")
    return metrics

def submit_metrics(metrics):
    """Submit metrics to Datadog API."""
    timestamp = metrics['timestamp']
    hostname = 'monitoring-server'
    
    # Format metrics into Datadog's API format
    series = [
        {
            'metric': 'system.cpu.usage',
            'points': [[timestamp, metrics['cpu_usage']]],
            'type': 'gauge',
            'host': hostname
        },
        {
            'metric': 'system.memory.usage',
            'points': [[timestamp, metrics['memory_usage']]],
            'type': 'gauge',
            'host': hostname
        },
        {
            'metric': 'system.disk.usage',
            'points': [[timestamp, metrics['disk_usage']]],
            'type': 'gauge',
            'host': hostname
        }
    ]
    
    payload = {'series': series}
    
    headers = {
        'Content-Type': 'application/json',
        'DD-API-KEY': DATADOG_API_KEY,
        'DD-APPLICATION-KEY': DATADOG_APP_KEY
    }
    
    logging.info("Submitting metrics to Datadog...")
    response = requests.post(
        DATADOG_API_URL,
        headers=headers,
        data=json.dumps(payload)
    )
    
    return response

def main():
    """Main function to collect and submit metrics."""
    try:
        # Collect system metrics
        metrics = collect_metrics()
        
        # Submit metrics to Datadog
        response = submit_metrics(metrics)
        
        print(f"Response status code: {response.status_code}")
        
        # Check if submission was successful
        if response.status_code not in [200, 202]:
            raise Exception(f"Failed to submit metrics. Status: {response.status_code}, Response: {response.text}")
        
        logging.info("Metrics submitted successfully")
        
    except Exception as e:
        logging.error(f"Error in metrics submission: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()