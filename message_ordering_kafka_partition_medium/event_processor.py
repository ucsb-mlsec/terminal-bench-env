#!/usr/bin/env python3

import json
import random

NUM_PARTITIONS = 3

def assign_partition(event):
    """
    Assigns an event to a partition.
    BUG: Uses random assignment instead of user_id-based assignment.
    This causes events from the same user to be split across different partitions.
    """
    return random.randint(0, NUM_PARTITIONS - 1)

def process_events():
    """
    Reads events from test_events.json and distributes them to partitions.
    """
    # Read events from file
    with open('data/test_events.json', 'r') as f:
        events = json.load(f)
    
    # Initialize partitions
    partitions = {f"partition_{i}": [] for i in range(NUM_PARTITIONS)}
    
    # Distribute events to partitions
    for event in events:
        partition_id = assign_partition(event)
        partition_key = f"partition_{partition_id}"
        partitions[partition_key].append(event)
    
    # Write partitioned events to file
    with open('partitioned_events.json', 'w') as f:
        json.dump(partitions, f, indent=2)
    
    # Print summary
    print("Event Distribution Summary:")
    print("-" * 40)
    for partition_key in sorted(partitions.keys()):
        event_count = len(partitions[partition_key])
        print(f"{partition_key}: {event_count} events")
    print("-" * 40)
    print(f"Total events processed: {len(events)}")
    print(f"Events written to: partitioned_events.json")

if __name__ == "__main__":
    process_events()