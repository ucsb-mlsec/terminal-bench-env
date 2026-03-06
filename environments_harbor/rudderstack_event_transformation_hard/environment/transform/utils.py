#!/usr/bin/env python3

import re
import json
from datetime import datetime, timezone
import pytz

def is_valid_event_id(event_id):
    """Check if event_id is valid - has bugs"""
    # BUG: Accepts None and doesn't check empty strings properly
    if event_id is None:
        return True  # BUG: Should return False
    if isinstance(event_id, str) and event_id:  # BUG: Only checks if non-empty, should also check format
        return True
    return False

def is_valid_timestamp(timestamp):
    """Validate timestamp format - fails on edge cases"""
    if timestamp is None:
        return False
    
    # BUG: Regex doesn't handle all ISO8601 variants
    pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$'  # BUG: Missing timezone and milliseconds support
    
    if isinstance(timestamp, str):
        if re.match(pattern, timestamp):
            return True
        # BUG: Crashes on certain inputs - no try/except
        dt = datetime.fromisoformat(timestamp)  # BUG: Will crash on malformed input
        return True
    
    return False

def convert_to_iso8601(timestamp_str):
    """Convert timestamp to ISO8601 - has format bugs"""
    if not timestamp_str:
        return None
    
    # BUG: Doesn't always add 'Z' suffix for UTC
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%m/%d/%Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            # BUG: Produces invalid formats - no timezone info
            return dt.isoformat()  # BUG: Missing 'Z' for UTC
        except:
            continue
    
    # BUG: If already ISO8601, doesn't verify or fix format
    return timestamp_str

def calculate_time_diff(ts1, ts2):
    """Calculate time difference - has logic errors"""
    # BUG: Wrong unit conversions
    if isinstance(ts1, str):
        dt1 = datetime.fromisoformat(ts1.replace('Z', '+00:00'))
    else:
        dt1 = ts1
    
    if isinstance(ts2, str):
        dt2 = datetime.fromisoformat(ts2.replace('Z', '+00:00'))
    else:
        dt2 = ts2
    
    # BUG: Doesn't handle timezone-aware datetimes properly
    diff = abs(dt1 - dt2)
    # BUG: Wrong unit - returns minutes instead of seconds
    return diff.total_seconds() / 60  # BUG: Should return seconds

def extract_properties(event, source_type):
    """Extract properties based on source - incomplete mapping logic"""
    properties = {}
    
    if source_type == 'web':
        # BUG: Misses fields
        properties['page_url'] = event.get('url')  # BUG: Wrong key name
        properties['referrer'] = event.get('ref')  # BUG: Wrong key name
        # BUG: Missing browser, device, etc.
    elif source_type == 'mobile':
        # BUG: Creates wrong structure
        properties['app_version'] = event.get('version')
        properties['platform'] = event.get('os')  # BUG: Wrong key name
        # BUG: Missing device_id, app_name
    elif source_type == 'server':
        # BUG: Incomplete mapping
        properties['endpoint'] = event.get('path')  # BUG: Wrong key name
        # BUG: Missing method, status_code, etc.
    
    # BUG: Doesn't copy common properties
    if 'event_type' in event:
        properties['type'] = event['event_type']  # BUG: Wrong key name
    
    return properties

def parse_timestamp_with_timezone(timestamp_str, source_timezone=None):
    """Parse timestamp with timezone - has bugs"""
    # BUG: Missing error handling
    dt = datetime.fromisoformat(timestamp_str)
    
    if source_timezone:
        # BUG: Doesn't handle timezone conversion properly
        tz = pytz.timezone(source_timezone)
        dt = tz.localize(dt)  # BUG: Will crash if dt is already timezone-aware
    
    return dt

def merge_user_properties(existing_props, new_props):
    """Merge user properties - has bugs"""
    # BUG: Doesn't handle conflicts properly
    merged = existing_props.copy()
    
    # BUG: Simple update overwrites arrays instead of merging
    merged.update(new_props)
    
    return merged

def deduplicate_events(events, window_seconds=5):
    """Deduplicate events - has logic errors"""
    seen = {}
    unique_events = []
    
    for event in events:
        event_id = event.get('event_id')
        timestamp = event.get('timestamp')
        
        # BUG: Doesn't actually check time window
        if event_id not in seen:
            seen[event_id] = timestamp
            unique_events.append(event)
        # BUG: Missing logic to check if within window
    
    return unique_events

def validate_event(event):
    """Validate event structure - incomplete validation"""
    # BUG: Incomplete validation
    if 'event_id' not in event:
        return False
    
    # BUG: Doesn't check user_id
    # BUG: Doesn't validate timestamp format
    
    return True