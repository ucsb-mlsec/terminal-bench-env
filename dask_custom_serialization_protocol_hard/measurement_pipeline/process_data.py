#!/usr/bin/env python3

import json
import numpy as np
from dask.distributed import Client, wait
from measurement import Measurement
import serialization_handler

def process_measurement(measurement):
    """Process a measurement and return statistics."""
    return {
        'sensor_id': measurement.sensor_id,
        'mean': float(np.mean(measurement.readings)),
        'sum': float(np.sum(measurement.readings)),
        'std': float(np.std(measurement.readings)),
        'timestamp': measurement.timestamp
    }

def main():
    try:
        # Load test data
        with open('test_measurements.json', 'r') as f:
            data = json.load(f)
        
        # Create Measurement objects
        measurements = []
        for item in data:
            m = Measurement(
                sensor_id=item['sensor_id'],
                readings=np.array(item['readings'], dtype=float),
                timestamp=item['timestamp']
            )
            measurements.append(m)
        
        # Start Dask client
        client = Client(n_workers=2, threads_per_worker=1, processes=True)
        
        try:
            # Submit tasks to process measurements
            futures = [client.submit(process_measurement, m) for m in measurements]
            
            # Wait for all tasks to complete
            wait(futures)
            
            # Gather results
            results = [f.result() for f in futures]
            
            # Verify results
            all_passed = True
            for i, result in enumerate(results):
                original = measurements[i]
                
                # Verify sensor_id matches
                if result['sensor_id'] != original.sensor_id:
                    all_passed = False
                    break
                
                # Verify timestamp matches
                if result['timestamp'] != original.timestamp:
                    all_passed = False
                    break
                
                # Verify statistics are correct
                expected_mean = np.mean(original.readings)
                expected_sum = np.sum(original.readings)
                expected_std = np.std(original.readings)
                
                if not np.isclose(result['mean'], expected_mean):
                    all_passed = False
                    break
                
                if not np.isclose(result['sum'], expected_sum):
                    all_passed = False
                    break
                
                if not np.isclose(result['std'], expected_std):
                    all_passed = False
                    break
            
            if all_passed and len(results) == len(measurements):
                print('SUCCESS')
            else:
                print('FAILED')
        
        finally:
            client.close()
    
    except Exception as e:
        print(f'FAILED: {str(e)}')

if __name__ == '__main__':
    main()