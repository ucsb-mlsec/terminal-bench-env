#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

def read_shard_metadata():
    """Read shard metadata from config directory."""
    config_dir = Path('/workspace/config')
    shards = {}
    
    try:
        # Read shard configuration
        shard_config_file = config_dir / 'shards.json'
        if shard_config_file.exists():
            with open(shard_config_file, 'r') as f:
                shards = json.load(f)
        else:
            # Default shard configuration if file doesn't exist
            shards = {
                "shard-000001": {
                    "hash_key_start": "0",
                    "hash_key_end": "85070591730234615865843651857942052863",
                    "traffic_percentage": 42.5
                },
                "shard-000002": {
                    "hash_key_start": "85070591730234615865843651857942052864",
                    "hash_key_end": "170141183460469231731687303715884105727",
                    "traffic_percentage": 28.3
                },
                "shard-000003": {
                    "hash_key_start": "170141183460469231731687303715884105728",
                    "hash_key_end": "255211775190703847597530955573826158591",
                    "traffic_percentage": 18.7
                },
                "shard-000004": {
                    "hash_key_start": "255211775190703847597530955573826158592",
                    "hash_key_end": "340282366920938463463374607431768211455",
                    "traffic_percentage": 10.5
                }
            }
    except Exception as e:
        print(f"Error reading shard metadata: {e}", file=sys.stderr)
        sys.exit(1)
    
    return shards

def read_mock_data():
    """Read mock AWS SDK responses."""
    mock_dir = Path('/workspace/mock_data')
    mock_data = {}
    
    try:
        mock_file = mock_dir / 'stream_metrics.json'
        if mock_file.exists():
            with open(mock_file, 'r') as f:
                mock_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read mock data: {e}", file=sys.stderr)
    
    return mock_data

def display_shard_info(shards):
    """Display current shard information."""
    print("\n=== Current Shard Configuration ===")
    print(f"Total Shards: {len(shards)}")
    print("\nShard Details:")
    
    for shard_id, info in sorted(shards.items()):
        print(f"\n{shard_id}:")
        print(f"  Hash Key Range: {info['hash_key_start']} - {info['hash_key_end']}")
        print(f"  Traffic: {info['traffic_percentage']}%")

def validate_hash_key_ranges(shards_list):
    """Validate that hash key ranges are sequential with no gaps or overlaps."""
    sorted_shards = sorted(shards_list, key=lambda x: int(x['hash_key_start']))
    
    # Check first shard starts at 0
    if sorted_shards[0]['hash_key_start'] != "0":
        return False, "First shard must start at hash key 0"
    
    # Check last shard ends at max value
    max_hash = "340282366920938463463374607431768211455"
    if sorted_shards[-1]['hash_key_end'] != max_hash:
        return False, "Last shard must end at maximum hash key"
    
    # Check for gaps and overlaps
    for i in range(len(sorted_shards) - 1):
        current_end = int(sorted_shards[i]['hash_key_end'])
        next_start = int(sorted_shards[i + 1]['hash_key_start'])
        
        if current_end + 1 != next_start:
            return False, f"Gap or overlap detected between shards"
    
    return True, "Hash key ranges are valid"

def simulate_split(shards, shards_to_split):
    """Simulate splitting shards and calculate new traffic distribution."""
    new_shards = {}
    
    for shard_id, info in shards.items():
        if shard_id in shards_to_split:
            # Split this shard into two
            start = int(info['hash_key_start'])
            end = int(info['hash_key_end'])
            mid = start + (end - start) // 2
            
            # First half
            new_shard_id_1 = f"{shard_id}-child-1"
            new_shards[new_shard_id_1] = {
                "hash_key_start": str(start),
                "hash_key_end": str(mid),
                "traffic_percentage": info['traffic_percentage'] / 2
            }
            
            # Second half
            new_shard_id_2 = f"{shard_id}-child-2"
            new_shards[new_shard_id_2] = {
                "hash_key_start": str(mid + 1),
                "hash_key_end": str(end),
                "traffic_percentage": info['traffic_percentage'] / 2
            }
        else:
            # Keep this shard as is
            new_shards[shard_id] = info.copy()
    
    return new_shards

def validate_split_plan(shards, split_plan_path):
    """Validate the split plan against all requirements."""
    print("\n=== Validating Split Plan ===")
    
    if not split_plan_path.exists():
        print("ERROR: Split plan file not found")
        return False
    
    try:
        with open(split_plan_path, 'r') as f:
            plan = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in split plan: {e}")
        return False
    
    # Validate required fields
    required_fields = ['shards_to_split', 'total_shards_after', 'max_traffic_percentage']
    for field in required_fields:
        if field not in plan:
            print(f"ERROR: Missing required field: {field}")
            return False
    
    shards_to_split = plan['shards_to_split']
    total_shards_after = plan['total_shards_after']
    max_traffic_percentage = plan['max_traffic_percentage']
    
    print(f"\nPlan Summary:")
    print(f"  Shards to split: {shards_to_split}")
    print(f"  Total shards after: {total_shards_after}")
    print(f"  Max traffic percentage: {max_traffic_percentage}%")
    
    # Requirement 1: Total shards <= 8
    if total_shards_after > 8:
        print(f"ERROR: Total shards after splitting ({total_shards_after}) exceeds maximum of 8")
        return False
    print("✓ Total shard count constraint met (≤8)")
    
    # Requirement 2: Only split shards with >35% traffic
    for shard_id in shards_to_split:
        if shard_id not in shards:
            print(f"ERROR: Shard {shard_id} does not exist")
            return False
        if shards[shard_id]['traffic_percentage'] <= 35:
            print(f"ERROR: Shard {shard_id} has {shards[shard_id]['traffic_percentage']}% traffic (≤35%)")
            return False
    print("✓ Only high-traffic shards (>35%) are being split")
    
    # Calculate expected shard count
    current_shards = len(shards)
    expected_total = current_shards + len(shards_to_split)
    if total_shards_after != expected_total:
        print(f"ERROR: Expected {expected_total} shards after splitting, but plan states {total_shards_after}")
        return False
    print(f"✓ Shard count calculation correct ({expected_total} shards)")
    
    # Simulate the split and check traffic distribution
    new_shards = simulate_split(shards, shards_to_split)
    
    # Verify hash key ranges
    shard_list = [info for info in new_shards.values()]
    valid, message = validate_hash_key_ranges(shard_list)
    if not valid:
        print(f"ERROR: Hash key range validation failed: {message}")
        return False
    print("✓ Hash key ranges are sequential with no gaps")
    
    # Requirement 3: No shard > 30% traffic after split
    actual_max_traffic = max(info['traffic_percentage'] for info in new_shards.values())
    if actual_max_traffic > 30.0:
        print(f"ERROR: Maximum traffic percentage after split is {actual_max_traffic}% (>30%)")
        return False
    print(f"✓ No single shard exceeds 30% traffic (max: {actual_max_traffic}%)")
    
    # Verify reported max traffic matches calculation
    if abs(max_traffic_percentage - actual_max_traffic) > 0.1:
        print(f"WARNING: Reported max traffic ({max_traffic_percentage}%) differs from calculated ({actual_max_traffic}%)")
    
    # Requirement 4: Maintain minimum capacity (4 shards worth)
    if total_shards_after < 4:
        print(f"ERROR: Total shards after splitting ({total_shards_after}) is less than minimum of 4")
        return False
    print("✓ Minimum capacity maintained (≥4 shards)")
    
    print("\n=== Post-Split Shard Distribution ===")
    for shard_id, info in sorted(new_shards.items()):
        print(f"{shard_id}: {info['traffic_percentage']:.1f}% traffic")
    
    print("\n✓ ALL VALIDATION CHECKS PASSED")
    return True

def main():
    """Main function to monitor and validate Kinesis stream."""
    print("=== Kinesis Stream Monitor ===")
    
    # Read current configuration
    shards = read_shard_metadata()
    mock_data = read_mock_data()
    
    # Display current state
    display_shard_info(shards)
    
    # Check for split plan
    split_plan_path = Path('/workspace/solution/split_plan.json')
    
    if split_plan_path.exists():
        validation_passed = validate_split_plan(shards, split_plan_path)
        if validation_passed:
            print("\n✅ Split plan validation SUCCESSFUL")
            sys.exit(0)
        else:
            print("\n❌ Split plan validation FAILED")
            sys.exit(1)
    else:
        print("\n⚠️  No split plan found at /workspace/solution/split_plan.json")
        print("Please create a split plan to address the throughput bottleneck.")
        sys.exit(1)

if __name__ == "__main__":
    main()