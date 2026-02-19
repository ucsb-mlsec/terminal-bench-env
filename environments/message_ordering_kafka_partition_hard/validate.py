#!/usr/bin/env python3

import json
import pandas as pd
import sys
from pathlib import Path

def validate_partition_assignment():
    """
    Validate that partition assignments meet all requirements.
    """
    success = True
    
    # Define file paths
    output_json_path = Path('/workspace/output.json')
    events_csv_path = Path('/workspace/partitioner/events.csv')
    
    print("Starting validation...\n")
    
    # Check if files exist
    if not output_json_path.exists():
        print(f"✗ FAIL: Output file not found at {output_json_path}")
        return False
    
    if not events_csv_path.exists():
        print(f"✗ FAIL: Events CSV not found at {events_csv_path}")
        return False
    
    # Read output.json
    try:
        with open(output_json_path, 'r') as f:
            output_data = json.load(f)
    except Exception as e:
        print(f"✗ FAIL: Could not read output.json: {e}")
        return False
    
    # Read events.csv
    try:
        events_df = pd.read_csv(events_csv_path)
    except Exception as e:
        print(f"✗ FAIL: Could not read events.csv: {e}")
        return False
    
    # Validation 1: Check total_events
    expected_total = len(events_df)
    actual_total = output_data.get('total_events', 0)
    
    if actual_total == expected_total:
        print(f"✓ PASS: Total events: {actual_total} (expected {expected_total})")
    else:
        print(f"✗ FAIL: Total events: {actual_total} (expected {expected_total})")
        success = False
    
    # Validation 2: Check partitions_used
    expected_partitions = 8
    actual_partitions = output_data.get('partitions_used', 0)
    
    if actual_partitions == expected_partitions:
        print(f"✓ PASS: Partitions used: {actual_partitions} (expected {expected_partitions})")
    else:
        print(f"✗ FAIL: Partitions used: {actual_partitions} (expected {expected_partitions})")
        success = False
    
    # Validation 3: Check ordering_violations
    expected_violations = 0
    actual_violations = output_data.get('ordering_violations', -1)
    
    if actual_violations == expected_violations:
        print(f"✓ PASS: Ordering violations: {actual_violations} (expected {expected_violations})")
    else:
        print(f"✗ FAIL: Ordering violations: {actual_violations} (expected {expected_violations})")
        success = False
    
    # Validation 4: Verify each customer_id consistently maps to a single partition
    # This requires reading the partition assignments from a file
    # We'll check if a partition_assignments.csv exists or recreate the logic
    
    partition_file = Path('/workspace/partitioner/partition_assignments.csv')
    if partition_file.exists():
        try:
            partition_df = pd.read_csv(partition_file)
            
            # Check each customer has only one unique partition
            customer_partition_counts = partition_df.groupby('customer_id')['partition'].nunique()
            customers_with_multiple_partitions = customer_partition_counts[customer_partition_counts > 1]
            
            if len(customers_with_multiple_partitions) == 0:
                print(f"✓ PASS: All customers assigned to single partition")
            else:
                print(f"✗ FAIL: {len(customers_with_multiple_partitions)} customers split across multiple partitions")
                success = False
            
            # Check that partitions are in valid range (0-7)
            invalid_partitions = partition_df[(partition_df['partition'] < 0) | (partition_df['partition'] > 7)]
            if len(invalid_partitions) == 0:
                print(f"✓ PASS: All partitions in valid range (0-7)")
            else:
                print(f"✗ FAIL: {len(invalid_partitions)} events assigned to invalid partitions")
                success = False
            
            # Check no events are duplicated
            if len(partition_df) == len(events_df):
                print(f"✓ PASS: No events duplicated or missing")
            else:
                print(f"✗ FAIL: Event count mismatch (partition file: {len(partition_df)}, CSV: {len(events_df)})")
                success = False
                
        except Exception as e:
            print(f"✗ FAIL: Could not validate partition assignments: {e}")
            success = False
    else:
        print(f"⚠ WARNING: partition_assignments.csv not found, skipping detailed validation")
    
    # Final summary
    print("\n" + "="*50)
    if success:
        print("✓ All validations passed!")
        return True
    else:
        print("✗ Validation failed!")
        return False

if __name__ == "__main__":
    result = validate_partition_assignment()
    sys.exit(0 if result else 1)