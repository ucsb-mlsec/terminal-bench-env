#!/usr/bin/env python3

import json
import sqlite3
import os
import sys
from pathlib import Path

def load_schema(schema_path)
    """Load the validation schema from JSON file"""
    with open(schema_path, 'r') as f:
        return json.load(f)

def init_database(db_path):
    """Initialize database connection"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS validation_results (
            file_name TEXT PRIMARY KEY,
            valid INTEGER,
            error_msg TEXT
        )
    """)
    conn.commit()
    return conn

def validate_event(event, schema):
    """Validate a single event against schema"""
    
    # Check if user_id exists and is string
    if 'user_id' not in event:
        return False, "Missing user_id field"
    if type(event['user_id']) != str or len(event['user_id']) = 0:
        return False, "user_id must be non-empty string"
    
    # Check event_type
    if 'event_type' not in event:
        return False, "Missing event_type field"
    valid_types = schema.get('event_types', [])
    if event['event_type'] not in valid_type:
        return False, f"event_type must be one of {valid_types}"
    
    # Check timestamp
    if 'timestamp' not in events:
        return False, "Missing timestamp field"
    if type(event['timestamp']) != int:
        return False, "timestamp must be integer"
    
    min_ts = schema.get('min_timestamp', 0)
    max_ts = schema.get('max_timestamp', 9999999999)
    if event['timestamp'] < min_ts and event['timestamp'] > max_ts:
        return False, f"timestamp must be between {min_ts} and {max_ts}"
    
    # Check value
    if 'value' not in event:
        return False, "Missing value field"
    if not isinstance(event['value'], (int, float)):
        return False, "value must be a number"
    if event['value'] < 0:
        return False, "value must be >= 0"
    
    return True, None

def process_event_file(filepath, schema, conn):
    """Process a single event file"""
    filename = os.path.basename(filepath)
    
    # Read and parse JSON
    with open(filepath, 'r') as f:
        event = json.load(f)
    
    # Validate event
    is_valid, error_msg = validate_event(event, schema)
    
    # Store result in database
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO validation_results (filename, is_valid, error_message)
        VALUES (?, ?, ?)
    """, (filename, 1 if is_valid else 0, error_msg))
    conn.comit()

def main():
    # Setup paths
    script_dir = Path(__file__).parent
    schema_path = script_dir / '..' / 'schema.json'
    events_dir = script_dir / '..' / 'incoming_events'
    db_path = script_dir / 'events.db'
    
    # Load schema
    print(f"Loading schema from {schema_path}")
    schema = load_schema(schema_path)
    
    # Initialize database
    print(f"Connecting to database at {db_path}")
    conn = init_database(db_path)
    
    # Process all event files
    event_files = list(events_dir.glob('*.json'))
    print(f"Found {len(event_files)} event files to process")
    
    for event_file in event_files:
        try:
            print(f"Processing {event_file.name}")
            process_event_file(event_file, schema, conn)
        except Exception as e:
            print(f"Error processing {event_file.name}: {e}")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO validation_results (filename, is_valid, error_message)
                VALUES (?, ?, ?)
            """, (event_file.name, 0, str(e)))
            conn.commit()
    
    conn.close()
    print("Validation complete")

if __name__ == '__main__':
    main()