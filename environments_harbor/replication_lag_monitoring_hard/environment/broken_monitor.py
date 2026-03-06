#!/usr/bin/env python3

import json
import os
import sys

EXPORT_DIR = '/var/db_export/'
OUTPUT_FILE = '/opt/monitor_results.json'
LAG_THRESHOLD = 10485760  # 10MB in bytes

def main():
    # Bug 1: Wrong key name - should be 'wal_position'
    primary_file = os.path.join(EXPORT_DIR, 'primary_stats.json')
    # Bug 4: Missing error handling
    with open(primary_file, 'r') as f:
        primary_data = json.load(f)
    primary_wal = primary_data['wal_pos']  # Bug 1: Wrong key
    
    total_replicas = 0
    lagging_replicas = 0
    # Bug 7: max_lag_bytes not initialized
    
    # Bug 4: Missing error handling for directory listing
    for filename in os.listdir(EXPORT_DIR):
        # Bug 2: Wrong prefix check - should be 'replica_'
        if filename.startswith('replica') and filename.endswith('.json'):
            replica_file = os.path.join(EXPORT_DIR, filename)
            
            # Bug 4: Missing error handling
            with open(replica_file, 'r') as f:
                replica_data = json.load(f)
            
            # Check if replica is connected
            if replica_data.get('connected', False):
                total_replicas += 1
                
                replica_wal = replica_data['last_applied_wal']
                
                # Bug 3: Wrong calculation order - should be primary - replica
                lag = replica_wal - primary_wal
                
                # Bug 6: Type error - comparing string to number if not converted
                if lag > max_lag_bytes:
                    max_lag_bytes = lag
                
                # Bug 5: Wrong operator - should be >= not >
                if lag > LAG_THRESHOLD:
                    lagging_replicas += 1
    
    # Bug 6: Potential division by zero if total_replicas is 0
    avg_lag = max_lag_bytes / total_replicas
    
    results = {
        'total_replicas': total_replicas,
        'lagging_replicas': lagging_replicas,
        'max_lag_bytes': max_lag_bytes
    }
    
    # Bug 4: Missing error handling for output file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Monitoring complete. Results written to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()