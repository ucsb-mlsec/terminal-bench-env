#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
from pathlib import Path
from collections import defaultdict

def parse_args():
    """Parse command line arguments for worker configuration."""
    parser = argparse.ArgumentParser(description='Simulate distributed training worker initialization')
    parser.add_argument('--worker-config', type=str, help='Path to worker YAML configuration file')
    return parser.parse_args()

def load_config(config_path):
    """Load YAML configuration file.
    
    Args:
        config_path: Path to the YAML file
        
    Returns:
        Dictionary containing configuration data
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return None

def load_env_vars(env_file):
    """Load environment variables from file.
    
    Args:
        env_file: Path to environment variable file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error loading environment variables from {env_file}: {e}")
    return env_vars

def validate_rank(rank, world_size, seen_ranks):
    """Validate that a worker's rank is unique and within valid range.
    
    Args:
        rank: The rank to validate
        world_size: Total number of workers
        seen_ranks: Set of ranks already seen
        
    Returns:
        True if rank is valid and unique, False otherwise
    """
    # Check if rank is within valid range [0, world_size-1]
    if rank < 0 or rank >= world_size:
        print(f"  ERROR: Rank {rank} is out of range [0, {world_size-1}]")
        return False
    
    # Check if rank is unique (not already assigned)
    if rank in seen_ranks:
        print(f"  ERROR: Duplicate rank {rank} detected!")
        return False
    
    return True

def validate_world_size(world_sizes, expected_world_size):
    """Validate that all workers report consistent world_size.
    
    Args:
        world_sizes: Dictionary mapping worker to their reported world_size
        expected_world_size: Expected world_size from environment
        
    Returns:
        True if all world_sizes are consistent and match expected, False otherwise
    """
    if not world_sizes:
        print("  ERROR: No world_size values found")
        return False
    
    # Check if all workers report the same world_size
    unique_sizes = set(world_sizes.values())
    if len(unique_sizes) > 1:
        print(f"  ERROR: Workers report inconsistent world_size values: {unique_sizes}")
        for worker, size in world_sizes.items():
            print(f"    {worker}: world_size={size}")
        return False
    
    # Check if reported world_size matches expected from environment
    reported_size = list(unique_sizes)[0]
    if reported_size != expected_world_size:
        print(f"  ERROR: Reported world_size ({reported_size}) does not match expected ({expected_world_size})")
        return False
    
    return True

def simulate_initialization():
    """Simulate distributed training initialization for all workers.
    
    Loads configurations, validates rank assignments and world_size consistency.
    Returns True if all validations pass, False otherwise.
    """
    print("=" * 60)
    print("Starting Distributed Training Initialization Simulation")
    print("=" * 60)
    
    # Load environment variables
    env_file = Path('/workspace/env/worker_env.txt')
    env_vars = load_env_vars(env_file)
    
    if 'WORKER_COUNT' not in env_vars:
        print("ERROR: WORKER_COUNT not found in environment variables")
        return False
    
    try:
        expected_world_size = int(env_vars['WORKER_COUNT'])
    except ValueError:
        print(f"ERROR: WORKER_COUNT '{env_vars['WORKER_COUNT']}' is not a valid integer")
        return False
    
    print(f"\nExpected world_size from environment: {expected_world_size}")
    
    # Load cluster configuration
    cluster_config_path = Path('/workspace/config/cluster.yaml')
    cluster_config = load_config(cluster_config_path)
    
    if not cluster_config:
        print("ERROR: Failed to load cluster configuration")
        return False
    
    # Find all worker configuration files
    config_dir = Path('/workspace/config')
    worker_configs = sorted(config_dir.glob('worker*.yaml'))
    
    if not worker_configs:
        print("ERROR: No worker configuration files found")
        return False
    
    print(f"Found {len(worker_configs)} worker configuration files")
    print()
    
    # Track ranks and world_sizes across all workers
    seen_ranks = set()
    world_sizes = {}
    validation_passed = True
    
    # Simulate initialization for each worker
    for worker_config_path in worker_configs:
        worker_name = worker_config_path.stem
        print(f"Initializing {worker_name}...")
        
        worker_config = load_config(worker_config_path)
        if not worker_config:
            print(f"  ERROR: Failed to load {worker_name} configuration")
            validation_passed = False
            continue
        
        # Extract rank and world_size from worker configuration
        rank = worker_config.get('rank')
        world_size = worker_config.get('world_size')
        
        if rank is None:
            print(f"  ERROR: rank not specified in {worker_name}")
            validation_passed = False
            continue
        
        if world_size is None:
            print(f"  ERROR: world_size not specified in {worker_name}")
            validation_passed = False
            continue
        
        print(f"  Rank: {rank}, World Size: {world_size}")
        
        # Validate rank assignment
        if not validate_rank(rank, world_size, seen_ranks):
            validation_passed = False
        else:
            seen_ranks.add(rank)
            print(f"  ✓ Rank {rank} is valid and unique")
        
        # Store world_size for consistency check
        world_sizes[worker_name] = world_size
        print()
    
    # Validate world_size consistency across all workers
    print("Validating world_size consistency...")
    if not validate_world_size(world_sizes, expected_world_size):
        validation_passed = False
    else:
        print(f"  ✓ All workers report consistent world_size: {expected_world_size}")
    
    # Check if we have the correct number of workers
    print(f"\nValidating worker count...")
    if len(seen_ranks) != expected_world_size:
        print(f"  ERROR: Expected {expected_world_size} workers but found {len(seen_ranks)} unique ranks")
        validation_passed = False
    else:
        print(f"  ✓ Correct number of workers: {expected_world_size}")
    
    # Check if all ranks are sequential
    print(f"\nValidating rank sequence...")
    expected_ranks = set(range(expected_world_size))
    if seen_ranks != expected_ranks:
        missing = expected_ranks - seen_ranks
        if missing:
            print(f"  ERROR: Missing ranks: {sorted(missing)}")
        validation_passed = False
    else:
        print(f"  ✓ All ranks [0, {expected_world_size-1}] are assigned")
    
    return validation_passed

def main():
    """Main entry point for the simulation script."""
    args = parse_args()
    
    # If specific worker config provided, we're testing individual worker
    # Otherwise, run full simulation
    if args.worker_config:
        print(f"Testing individual worker config: {args.worker_config}")
        # This mode could be used for debugging individual workers
        # For now, we'll run the full simulation
    
    # Run the full initialization simulation
    success = simulate_initialization()
    
    print()
    print("=" * 60)
    if success:
        print("✓ VALIDATION PASSED: All workers initialized correctly")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ VALIDATION FAILED: Configuration errors detected")
        print("=" * 60)
        print("\nSummary of issues:")
        print("- Check that all worker configs have unique ranks [0, N-1]")
        print("- Verify that all workers report the same world_size")
        print("- Ensure WORKER_COUNT in environment matches actual worker count")
        print("- Confirm cluster.yaml has correct master address and port")
        sys.exit(1)

if __name__ == '__main__':
    main()