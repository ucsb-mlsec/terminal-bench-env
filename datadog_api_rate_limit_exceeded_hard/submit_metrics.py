#!/usr/bin/env python3

# Naive metrics submission script with NO rate limiting
# This script will fail when hitting API rate limits

import requests
import json
import sys

DATADOG_API_URL = 'https://api.datadoghq.com/api/v1/series'
DATADOG_API_KEY = 'fake_api_key_12345'

def submit_metric(metric_data):
    """
    Submit a single metric to Datadog.
    NO rate limiting - will fail when API limits are exceeded.
    NO retry logic - fails immediately on errors.
    """
    # Create Datadog payload format
    payload = {
        'series': [{
            'metric': metric_data['metric'],
            'points': [[metric_data['timestamp'], metric_data['value']]],
            'host': metric_data['host'],
            'type': 'gauge'
        }]
    }
    
    # Simple POST with no error handling
    headers = {
        'Content-Type': 'application/json',
        'DD-API-KEY': DATADOG_API_KEY
    }
    
    try:
        response = requests.post(DATADOG_API_URL, json=payload, headers=headers)
        # Basic status code check only
        return 200 <= response.status_code < 300
    except Exception:
        return False

if __name__ == '__main__':
    # No argument validation
    if len(sys.argv) < 2:
        print("Usage: submit_metrics.py <metrics_file.jsonl>")
        sys.exit(1)
    
    metrics_file = sys.argv[1]
    
    # Load all metrics at once - no memory management
    with open(metrics_file, 'r') as f:
        for line in f:
            metric_data = json.loads(line.strip())
            # Submit immediately with no rate limiting
            # This will crash when we hit rate limits
            result = submit_metric(metric_data)
            if not result:
                print(f"Failed to submit metric: {metric_data['metric']}")
                # No retry, just fail and move on