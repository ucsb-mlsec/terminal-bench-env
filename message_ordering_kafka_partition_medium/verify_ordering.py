#!/usr/bin/env python3

import json
import sys
from datetime import datetime
from collections import defaultdict

def load_partitioned_events():
    """Load the partitioned events from JSON file"""
    try:
        with open('partitioned_events.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("ERROR: partitioned_events.json not found. Run event_processor.py first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("ERROR: Invalid JSON in partitioned_events.json")
        sys.exit(1)

def extract_all_events(partitioned_data):
    """Extract all events from all partitions"""
    all_events = []
    for partition_id, events in partitioned_data.items():
        for event in events:
            event['partition'] = partition_id
            all_events.append(event)
    return all_events

def group_events_by_user(events):
    """Group events by user_id"""
    user_events = defaultdict(list)
    for event in events:
        user_events[event['user_id']].append(event)
    return user_events

def check_user_in_multiple_partitions(user_events):
    """Check if any user appears in multiple partitions"""
    violations = []
    for user_id, events in user_events.items():
        partitions = set(event['partition'] for event in events)
        if len(partitions) > 1:
            violations.append((user_id, sorted(partitions)))
    return violations

def verify_chronological_order(user_events):
    """Verify that each user's events are in chronological order"""
    violations = []
    
    for user_id, events in user_events.items():
        # Sort events by timestamp to check ordering
        sorted_events = sorted(events, key=lambda e: datetime.fromisoformat(e['timestamp']))
        
        # Check if original order matches sorted order
        original_timestamps = [datetime.fromisoformat(e['timestamp']) for e in events]
        sorted_timestamps = [datetime.fromisoformat(e['timestamp']) for e in sorted_events]
        
        if original_timestamps != sorted_timestamps:
            violations.append(user_id)
    
    return violations

def main():
    print("Loading partitioned events...")
    partitioned_data = load_partitioned_events()
    
    print("Extracting events from all partitions...")
    all_events = extract_all_events(partitioned_data)
    
    print("Grouping events by user...")
    user_events = group_events_by_user(all_events)
    
    print(f"\nFound {len(user_events)} unique users across {len(partitioned_data)} partitions")
    
    # Print user distribution
    for partition_id in sorted(partitioned_data.keys()):
        users_in_partition = set(e['user_id'] for e in partitioned_data[partition_id])
        print(f"Partition {partition_id}: {len(users_in_partition)} users, {len(partitioned_data[partition_id])} events")
    
    # Check for users in multiple partitions
    print("\nChecking for users in multiple partitions...")
    multi_partition_violations = check_user_in_multiple_partitions(user_events)
    
    if multi_partition_violations:
        print("ERROR: Found users in multiple partitions:")
        for user_id, partitions in multi_partition_violations:
            print(f"  User {user_id} found in partitions: {partitions}")
        sys.exit(1)
    
    # Verify chronological ordering
    print("\nVerifying chronological ordering for each user...")
    ordering_violations = verify_chronological_order(user_events)
    
    if ordering_violations:
        print("ERROR: Found ordering violations:")
        for user_id in ordering_violations:
            print(f"  User {user_id} has out-of-order events")
        sys.exit(1)
    
    print("\nSUCCESS: All users maintain chronological ordering")
    print("All events for each user are in the same partition and properly ordered")
    sys.exit(0)

if __name__ == "__main__":
    main()