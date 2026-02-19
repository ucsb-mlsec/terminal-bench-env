#!/usr/bin/env python3

import os
import sys
import json
import time
import random

PARTITIONS_DIR = '/workspace/partitions'
CONSUMER_GROUP_DIR = '/workspace/consumer_group'
RESULTS_FILE = '/workspace/results.txt'

def acquire_partition(partition_id):
    """
    Attempts to claim a partition by creating a lock file.
    BUG: Has a race condition - checks existence then creates file non-atomically.
    """
    lock_file = os.path.join(CONSUMER_GROUP_DIR, f'partition_{partition_id}.lock')
    
    if not os.path.exists(lock_file):
        # Race condition window - multiple consumers can pass this check
        time.sleep(random.uniform(0.01, 0.1))
        with open(lock_file, 'w') as f:
            f.write('locked')
        return True
    return False

def release_partition(partition_id):
    """
    Releases a partition by removing the lock file.
    """
    lock_file = os.path.join(CONSUMER_GROUP_DIR, f'partition_{partition_id}.lock')
    if os.path.exists(lock_file):
        os.remove(lock_file)

def consume_partition(partition_id):
    """
    Reads and parses messages from a partition file.
    Returns a list of parsed message objects.
    """
    partition_file = os.path.join(PARTITIONS_DIR, f'partition_{partition_id}.txt')
    messages = []
    
    if not os.path.exists(partition_file):
        return messages
    
    with open(partition_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    message = json.loads(line)
                    messages.append(message)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse message in partition {partition_id}")
    
    return messages

def main():
    """
    Main consumer logic that attempts to acquire partitions and consume messages.
    """
    # Ensure consumer group directory exists
    os.makedirs(CONSUMER_GROUP_DIR, exist_ok=True)
    
    # Discover available partitions
    partition_files = []
    if os.path.exists(PARTITIONS_DIR):
        for filename in os.listdir(PARTITIONS_DIR):
            if filename.startswith('partition_') and filename.endswith('.txt'):
                partition_id = filename.replace('partition_', '').replace('.txt', '')
                try:
                    partition_files.append(int(partition_id))
                except ValueError:
                    continue
    
    partition_files.sort()
    
    acquired_partitions = []
    all_messages = []
    
    # Try to acquire and consume partitions
    for partition_id in partition_files:
        if acquire_partition(partition_id):
            acquired_partitions.append(partition_id)
            messages = consume_partition(partition_id)
            all_messages.extend(messages)
            print(f"Consumer acquired and processed partition {partition_id}: {len(messages)} messages")
        else:
            print(f"Partition {partition_id} already locked by another consumer")
    
    # Write results
    with open(RESULTS_FILE, 'w') as f:
        f.write(f"consumed_count={len(all_messages)}\n")
        partitions_str = ','.join(map(str, acquired_partitions))
        f.write(f"partitions={partitions_str}\n")
    
    print(f"Total messages consumed: {len(all_messages)}")
    print(f"Partitions processed: {acquired_partitions}")
    
    # Simulate some processing time
    time.sleep(random.uniform(0.1, 0.3))
    
    # Release partitions
    for partition_id in acquired_partitions:
        release_partition(partition_id)

if __name__ == '__main__':
    main()