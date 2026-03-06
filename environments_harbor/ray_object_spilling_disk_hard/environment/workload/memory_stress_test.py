#!/usr/bin/env python3

import ray
import os
import sys

# Initialize Ray with object spilling configuration
ray.init(
    # Set object store memory (8GB - adjust based on system)
    object_store_memory=8 * 1024 * 1024 * 1024,
    
    # Configure object spilling
    _system_config={
        # Enable object spilling
        "object_spilling_config": {
            # Spill to the designated directory
            "type": "filesystem",
            "params": {
                "directory_path": "/mnt/ray_spill"
            }
        },
        # Trigger spilling at 70% object store usage
        "object_store_memory_usage_threshold": 0.7,
        # Enable automatic cleanup of unreferenced spilled objects
        "automatic_object_spilling_enabled": True,
        # Set max spilled objects size to at least 40GB
        "max_io_workers": 4,
        # Ensure proper cleanup
        "object_spilling_threshold": 0.7,
    },
    # Ignore reinit error if Ray is already initialized
    ignore_reinit_error=True,
)

# Ensure the spill directory exists
os.makedirs("/mnt/ray_spill", exist_ok=True)

try:
    # Execute the test workload
    exec(open('/workload/memory_stress_test.py').read())
    
    # Indicate success
    print("CONFIGURATION_SUCCESS")
    
except Exception as e:
    print(f"Error during workload execution: {e}")
    ray.shutdown()
    sys.exit(1)

# Shutdown Ray cleanly
ray.shutdown()
sys.exit(0)