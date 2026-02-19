#!/usr/bin/env python3

import json
from datetime import datetime, timedelta
from collections import defaultdict
import os

def parse_timestamp(ts_str):
    """Parse ISO 8601 timestamp string to datetime object"""
    return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

def get_hour_window(dt):
    """Get the start of the hour window for a given datetime"""
    return dt.replace(minute=0, second=0, microsecond=0)

def process_events_with_watermark(input_file, output_file):
    """
    Process events with proper event-time windowing and watermark handling.
    
    - Events are assigned to windows based on event_timestamp
    - Watermark tracks progress through event time
    - Late events up to 10 minutes are included in correct windows
    - Windows finalized only when watermark advances past window_end + lateness
    """
    
    # Configuration
    ALLOWED_LATENESS = timedelta(minutes=10)
    WINDOW_SIZE = timedelta(hours=1)
    
    # Read all events first
    events = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    
    # Sort events by arrival_timestamp to simulate stream processing
    events.sort(key=lambda e: parse_timestamp(e['arrival_timestamp']))
    
    # Data structures for windowing
    window_counts = defaultdict(int)  # window_start -> count
    finalized_windows = {}  # window_start -> final_count
    watermark = None  # Current watermark (event time)
    
    for event in events:
        event_time = parse_timestamp(event['event_timestamp'])
        arrival_time = parse_timestamp(event['arrival_timestamp'])
        
        # Update watermark: watermark = max event_time seen so far minus some buffer
        # In real systems, watermark strategy varies; here we use event_time directly
        # and track the maximum event time seen
        if watermark is None:
            watermark = event_time
        else:
            watermark = max(watermark, event_time)
        
        # Determine which window this event belongs to
        window_start = get_hour_window(event_time)
        window_end = window_start + WINDOW_SIZE
        
        # Check if event is too late (beyond allowed lateness)
        lateness_threshold = window_end + ALLOWED_LATENESS
        
        if watermark <= lateness_threshold:
            # Event is not too late, add it to the window
            window_counts[window_start] += 1
        elif event_time >= window_start and event_time < window_end:
            # Event belongs to this window but check if window is still open
            if window_start not in finalized_windows:
                # Window not yet finalized, include the event
                window_counts[window_start] += 1
        # else: event is too late and discarded
        
        # Finalize windows that are now beyond the watermark + allowed lateness
        windows_to_finalize = []
        for win_start in list(window_counts.keys()):
            if win_start not in finalized_windows:
                win_end = win_start + WINDOW_SIZE
                if watermark >= win_end + ALLOWED_LATENESS:
                    # Window can be finalized
                    finalized_windows[win_start] = window_counts[win_start]
                    windows_to_finalize.append(win_start)
        
        # Remove finalized windows from active window_counts
        for win_start in windows_to_finalize:
            del window_counts[win_start]
    
    # At the end of stream, finalize all remaining windows
    for win_start, count in window_counts.items():
        if win_start not in finalized_windows:
            finalized_windows[win_start] = count
    
    # Convert datetime keys to ISO 8601 strings for output
    hourly_counts = {}
    for win_start, count in finalized_windows.items():
        key = win_start.isoformat().replace('+00:00', 'Z')
        hourly_counts[key] = count
    
    # Sort by timestamp for cleaner output
    hourly_counts = dict(sorted(hourly_counts.items()))
    
    # Write output
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w') as f:
        json.dump({'hourly_counts': hourly_counts}, f, indent=2)
    
    return hourly_counts

if __name__ == '__main__':
    input_file = '/opt/stream_processor/data/sensor_events.json'
    output_file = '/opt/stream_processor/output/results.json'
    
    try:
        results = process_events_with_watermark(input_file, output_file)
        print(f"Processing complete. Results written to {output_file}")
        print(f"Processed {sum(results.values())} events across {len(results)} hour windows")
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
    except Exception as e:
        print(f"Error processing events: {e}")
        raise