#!/usr/bin/env python3

import subprocess
import sys
import os
import time
import re

def run_analytics():
    """Execute the analytics application and return its output."""
    try:
        result = subprocess.run(
            ['python', '/workspace/analytics_app.py'],
            capture_output=True,
            text=True,
            check=True,
            timeout=60
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running analytics: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        print("Analytics execution timed out")
        raise
    except Exception as e:
        print(f"Unexpected error running analytics: {e}")
        raise

def run_update_script():
    """Execute the data update script."""
    try:
        result = subprocess.run(
            ['bash', '/workspace/update_data.sh'],
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        print(f"Update script output: {result.stdout}")
        if result.stderr:
            print(f"Update script stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error running update script: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        print("Update script timed out")
        raise
    except Exception as e:
        print(f"Unexpected error running update script: {e}")
        raise

def extract_numeric_value(output):
    """Extract a numeric value from the analytics output."""
    # Try to find patterns like "Total: 12345" or "Count: 12345" or "Sum: 12345.67"
    patterns = [
        r'Total[:\s]+(\d+\.?\d*)',
        r'Count[:\s]+(\d+)',
        r'Sum[:\s]+(\d+\.?\d*)',
        r'Revenue[:\s]+(\d+\.?\d*)',
        r'Amount[:\s]+(\d+\.?\d*)',
        r'(\d+\.?\d+)'  # Fallback: any number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    # If no numeric value found, return hash of output as fallback
    return hash(output)

if __name__ == '__main__':
    try:
        print('Running initial analytics...')
        initial_output = run_analytics()
        print("Initial output:")
        print(initial_output)
        print("-" * 50)
        
        initial_value = extract_numeric_value(initial_output)
        print(f"Initial extracted value: {initial_value}")
        print("-" * 50)
        
        print('Updating data files...')
        run_update_script()
        
        print('Waiting for file system sync...')
        time.sleep(2)
        
        print('Running analytics after update...')
        updated_output = run_analytics()
        print("Updated output:")
        print(updated_output)
        print("-" * 50)
        
        updated_value = extract_numeric_value(updated_output)
        print(f"Updated extracted value: {updated_value}")
        print("-" * 50)
        
        # Compare values - they should be different if cache refresh works
        if initial_value != updated_value:
            print('TEST PASSED: Values changed after data update')
            print(f'Initial: {initial_value}, Updated: {updated_value}')
            sys.exit(0)
        else:
            print('TEST FAILED: Values did not change after data update')
            print(f'Both values: {initial_value}')
            print('This indicates cache is not being refreshed properly')
            sys.exit(1)
            
    except Exception as e:
        print(f'TEST FAILED with exception: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)