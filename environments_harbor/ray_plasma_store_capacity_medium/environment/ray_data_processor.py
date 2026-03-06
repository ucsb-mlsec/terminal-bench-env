#!/usr/bin/env python3
"""
Ray Data Processing Pipeline - Plasma Store Memory Issue Demonstration

This script demonstrates a Ray-based data processing pipeline that encounters
ObjectStoreFullError due to insufficient plasma store capacity. The pipeline
attempts to process large datasets by storing intermediate results in Ray's
plasma object store, but will fail with default Ray settings.

The plasma object store (shared memory) is where Ray stores objects that are
shared between tasks and actors. By default, Ray allocates approximately 30%
of system memory for the object store, which may be insufficient for large
data processing workloads.
"""

import ray
import numpy as np
import time

def process_large_dataset():
    """
    Simulates processing a large dataset by creating multiple large numpy arrays
    and storing them in Ray's plasma object store.
    
    This function attempts to store 10 chunks of 1GB each (total 10GB) in the
    plasma store. With default settings on systems with less than ~33GB RAM,
    this will trigger an ObjectStoreFullError.
    """
    object_refs = []
    num_chunks = 10
    
    print(f"\nAttempting to process {num_chunks} chunks of ~1GB each...")
    print("This requires at least 10GB of plasma store capacity.\n")
    
    for i in range(num_chunks):
        print(f"Processing chunk {i + 1} of {num_chunks}...")
        
        # Create a large numpy array (~1GB)
        # 125000000 float64 values = 125M * 8 bytes = 1GB
        large_array = np.random.rand(125000000)
        
        # Store in Ray's plasma object store
        # This is where the memory pressure occurs
        obj_ref = ray.put(large_array)
        object_refs.append(obj_ref)
        
        time.sleep(0.1)  # Small delay to observe progress
    
    print("\nAll chunks stored successfully in plasma store!")
    return object_refs


if __name__ == "__main__":
    try:
        # Initialize Ray without specifying object_store_memory
        # This uses default settings (~30% of system RAM)
        print("Initializing Ray cluster with default settings...")
        ray.init()
        
        print("Starting data processing pipeline...")
        print("=" * 60)
        
        # Attempt to process large dataset
        results = process_large_dataset()
        
        print("=" * 60)
        print("Processing completed successfully!")
        print(f"Total objects in store: {len(results)}")
        
    except ray.exceptions.ObjectStoreFullError as e:
        print("\n" + "=" * 60)
        print("ERROR: ObjectStoreFullError encountered!")
        print("=" * 60)
        print("\nThe plasma object store does not have enough capacity")
        print("to handle this workload. The pipeline requires at least")
        print("8-10GB of object store memory, but the current configuration")
        print("is insufficient.")
        print("\nTo fix this issue:")
        print("1. Run /tmp/fix_plasma_store.sh to reconfigure Ray")
        print("2. Re-run this script after reconfiguration")
        print(f"\nOriginal error: {str(e)}")
        
    finally:
        # Clean shutdown
        print("\nShutting down Ray...")
        ray.shutdown()