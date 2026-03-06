#!/usr/bin/env python3

import json
import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

class EventProcessor:
    def __init__(self):
        self.processed_events = []
        
    def load_events(self, filepath: str) -> List[Dict[str, Any]]:
        """Load events from JSON file - BUGGY: doesn't handle errors properly"""
        # Bug: No error handling for missing files or invalid JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    
    def normalize_timestamp(self, timestamp_str: str, timezone: str) -> Optional[str]:
        """Convert timestamp to UTC ISO8601 format - BUGGY: wrong timezone offsets"""
        try:
            # Bug: timezone offsets are incorrect
            if timezone == "PST":
                offset_hours = +8  # Should be -8, not +8
            elif timezone == "EST":
                offset_hours = +5  # Should be -5, not +5
            else:
                offset_hours = 0
            
            # Bug: doesn't handle different timestamp formats properly
            # Only tries one format
            dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            
            # Bug: applying offset in wrong direction
            utc_dt = dt - datetime.timedelta(hours=offset_hours)
            
            return utc_dt.isoformat() + "Z"
        except:
            # Bug: returns None silently on error instead of handling gracefully
            return None
    
    def deduplicate_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate events - BUGGY: ignores timestamp window, only checks event_id"""
        # Bug: Only checks event_id, doesn't consider 5-second window
        # Bug: Uses set which may drop valid events with same ID but different timestamps
        seen_ids = set()
        deduped = []
        
        for event in events:
            event_id = event.get('event_id')
            # Bug: doesn't check timestamp window at all
            if event_id not in seen_ids:
                seen_ids.add(event_id)
                deduped.append(event)
        
        return deduped
    
    def validate_event(self, event: Dict[str, Any]) -> bool:
        """Validate event has required fields - BUGGY: incomplete checks"""
        # Bug: doesn't check if user_id is empty string
        if 'event_id' in event:
            # Bug: allows None values
            if event.get('user_id'):
                # Bug: doesn't validate timestamp format or existence properly
                return True
        return False
    
    def merge_properties(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge user properties from multiple events - BUGGY: creates malformed structures"""
        # Bug: inefficient O(n²) nested loops
        merged = []
        
        for i, event in enumerate(events):
            props = event.get('properties', {})
            
            # Bug: overwrites entire properties dict instead of merging
            for j, other_event in enumerate(events):
                if i != j and event.get('user_id') == other_event.get('user_id'):
                    # Bug: just replaces properties instead of merging keys
                    props = other_event.get('properties', {})
            
            event['properties'] = props
            merged.append(event)
        
        return merged
    
    def enrich_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Add computed fields to event - BUGGY: missing null checks"""
        # Bug: doesn't check if properties exist before accessing
        event['enriched'] = True
        
        # Bug: will fail if properties don't exist
        event['property_count'] = len(event['properties'])
        
        return event
    
    def process_batch(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of events - BUGGY: very inefficient"""
        processed = []
        
        # Bug: sorts multiple times unnecessarily (O(n log n) repeated)
        for event in events:
            if self.validate_event(event):
                processed.append(event)
        
        # Bug: sorts before deduplication instead of after
        processed = sorted(processed, key=lambda x: x.get('timestamp', ''))
        
        # Bug: calls merge_properties which has O(n²) complexity
        processed = self.merge_properties(processed)
        
        # Bug: enriches before deduplication, wasting work
        for i in range(len(processed)):
            processed[i] = self.enrich_event(processed[i])
        
        # Bug: deduplication with wrong logic
        processed = self.deduplicate_events(processed)
        
        # Bug: sorts again unnecessarily
        processed = sorted(processed, key=lambda x: x.get('timestamp', ''))
        
        return processed
    
    def transform_web_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform web events to standard format"""
        transformed = []
        for event in events:
            # Bug: doesn't handle missing fields
            transformed_event = {
                'event_id': event['id'],
                'timestamp': self.normalize_timestamp(event['time'], event['tz']),
                'user_id': event['uid'],
                'properties': event.get('props', {})
            }
            transformed.append(transformed_event)
        return transformed
    
    def transform_mobile_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform mobile events to standard format"""
        transformed = []
        for event in events:
            # Bug: different field names not handled consistently
            transformed_event = {
                'event_id': event['event_id'],
                'timestamp': self.normalize_timestamp(event['timestamp'], event['timezone']),
                'user_id': event['user'],
                'properties': event.get('data', {})
            }
            transformed.append(transformed_event)
        return transformed
    
    def transform_server_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform server events to standard format"""
        transformed = []
        for event in events:
            transformed_event = {
                'event_id': event['evt_id'],
                'timestamp': self.normalize_timestamp(event['ts'], 'UTC'),
                'user_id': event['user_id'],
                'properties': event.get('attributes', {})
            }
            transformed.append(transformed_event)
        return transformed
    
    def run_pipeline(self):
        """Main pipeline execution - BUGGY: doesn't handle errors"""
        all_events = []
        
        # Bug: hardcoded paths, no error handling
        web_events = self.load_events('/workspace/raw_events/web_events.json')
        all_events.extend(self.transform_web_events(web_events))
        
        mobile_events = self.load_events('/workspace/raw_events/mobile_events.json')
        all_events.extend(self.transform_mobile_events(mobile_events))
        
        server_events = self.load_events('/workspace/raw_events/server_events.json')
        all_events.extend(self.transform_server_events(server_events))
        
        # Process all events
        processed = self.process_batch(all_events)
        
        # Bug: doesn't ensure output directory exists
        with open('/workspace/output/transformed_events.json', 'w') as f:
            json.dump(processed, f, indent=2)
        
        return processed

if __name__ == "__main__":
    processor = EventProcessor()
    processor.run_pipeline()