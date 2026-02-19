#!/usr/bin/env python3

import json
import time
import os
import glob
from pathlib import Path

CHECKPOINT_FILE = 'checkpoint.txt'
EVENTS_DIR = 'events/'
LOG_FILE = 'logs/errors.log'
READ_WINDOW_TIMEOUT = 300  # 5 minutes in seconds

def setup_logging():
    """Ensure log directory exists"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_error(message):
    """Write error message to log file"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

def read_checkpoint():
    """Read the last processed timestamp from checkpoint file"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            content = f.read().strip()
            return float(content) if content else 0.0
    return 0.0

def write_checkpoint(timestamp):
    """Write the current timestamp to checkpoint file"""
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(timestamp))

def get_event_files():
    """Get sorted list of event files"""
    pattern = os.path.join(EVENTS_DIR, 'events_*.json')
    files = glob.glob(pattern)
    # Sort by the numeric part of filename
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    return files

def process_record(record):
    """Process a single event record"""
    # Simulate some processing
    timestamp = record['timestamp']
    data = record['data']
    print(f"Processing record: timestamp={timestamp}, data={data}")
    return timestamp

def main():
    setup_logging()
    
    last_checkpoint = read_checkpoint()
    print(f"Starting from checkpoint: {last_checkpoint}")
    
    event_files = get_event_files()
    print(f"Found {len(event_files)} event files to process")
    
    last_read_time = time.time()
    records_processed = 0
    
    for event_file in event_files:
        print(f"\nProcessing file: {event_file}")
        
        # Inefficient delay between files
        time.sleep(15)
        
        if not os.path.exists(event_file):
            log_error(f"Event file not found: {event_file}")
            continue
        
        with open(event_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    record_timestamp = record['timestamp']
                    
                    # Skip already processed records
                    if record_timestamp <= last_checkpoint:
                        continue
                    
                    # Check if read window expired
                    current_time = time.time()
                    time_since_last_read = current_time - last_read_time
                    
                    if time_since_last_read > READ_WINDOW_TIMEOUT:
                        error_msg = f"read window expired: {time_since_last_read:.1f} seconds since last read"
                        log_error(error_msg)
                        print(f"ERROR: {error_msg}")
                    
                    # Process the record
                    processed_timestamp = process_record(record)
                    records_processed += 1
                    
                    # Update checkpoint
                    write_checkpoint(processed_timestamp)
                    last_checkpoint = processed_timestamp
                    last_read_time = current_time
                    
                    # Inefficient delay between records
                    time.sleep(10)
                    
                except json.JSONDecodeError as e:
                    log_error(f"JSON decode error in {event_file}: {e}")
                except Exception as e:
                    log_error(f"Error processing record in {event_file}: {e}")
    
    print(f"\nProcessing complete. Total records processed: {records_processed}")
    print(f"Final checkpoint: {last_checkpoint}")

if __name__ == '__main__':
    main()