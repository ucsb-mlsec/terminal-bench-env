#!/usr/bin/env python3

import json
import sys

MAX_HASH_KEY = 340282366920938463463374607431768211455

def load_traffic_data(filename):
    """Load traffic data from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filename}")
        sys.exit(1)

def calculate_split_point(start_hash, end_hash):
    """Calculate the midpoint for splitting a shard - BUG: uses float division instead of integer"""
    # BUG: Using / instead of // causes float result which breaks hash key calculations
    midpoint = (start_hash + end_hash) / 2
    return midpoint

def validate_split(parent_start, parent_end, child1_start, child1_end, child2_start, child2_end):
    """Validate that child shards properly cover parent range - BUG: wrong comparison operators"""
    # BUG: Using > instead of >= and < instead of <= causes boundary validation failures
    if child1_start != parent_start:
        return False
    if child2_end != parent_end:
        return False
    # BUG: Wrong operator - should be child1_end + 1 == child2_start
    if child1_end > child2_start:
        return False
    # BUG: Missing check for gap between child1_end and child2_start
    return True

def generate_split_plan(traffic_data):
    """Generate shard splitting plan based on traffic data"""
    shards = traffic_data.get('shards', [])
    splits = []
    shards_to_split = 0
    
    for shard in shards:
        if shard.get('needs_split', False):
            shards_to_split += 1
            shard_id = shard['shard_id']
            # BUG: Hash keys stored as strings but treated as integers without conversion
            start_hash = shard['start_hash_key']
            end_hash = shard['end_hash_key']
            
            # Calculate split point - will crash due to type mismatch
            split_point = calculate_split_point(start_hash, end_hash)
            
            # Define child shard ranges
            child1_start = start_hash
            child1_end = split_point
            child2_start = split_point + 1
            child2_end = end_hash
            
            # Validate the split
            is_valid = validate_split(start_hash, end_hash, 
                                     child1_start, child1_end,
                                     child2_start, child2_end)
            
            split_info = {
                'parent_shard': shard_id,
                'child1': {
                    'start': child1_start,
                    'end': child1_end
                },
                'child2': {
                    'start': child2_start,
                    'end': child2_end
                },
                'valid': is_valid
            }
            splits.append(split_info)
    
    # Calculate total shards after split
    original_shard_count = len(shards)
    # Each split adds 2 child shards and closes 1 parent (net +1)
    total_shards_after = original_shard_count + shards_to_split
    
    # Check if all splits are valid
    all_valid = all(s['valid'] for s in splits)
    
    return {
        'shards_to_split': shards_to_split,
        'total_shards_after': total_shards_after,
        'valid_plan': all_valid,
        'split_details': splits
    }

def main():
    """Main execution function"""
    input_file = '/home/user/kinesis/traffic_data.json'
    output_file = '/tmp/split_plan.json'
    
    # Load traffic data
    traffic_data = load_traffic_data(input_file)
    
    # Generate split plan
    plan = generate_split_plan(traffic_data)
    
    # BUG: Wrong output format - includes extra fields and wrong structure
    output = {
        'splits': plan['shards_to_split'],
        'total': plan['total_shards_after'],
        # BUG: Missing 'valid_plan' field in output
        'details': plan['split_details']
    }
    
    # Write output
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Split plan generated: {output_file}")
    print(f"Shards to split: {plan['shards_to_split']}")
    print(f"Total shards after: {plan['total_shards_after']}")
    print(f"Valid plan: {plan['valid_plan']}")

if __name__ == '__main__':
    main()