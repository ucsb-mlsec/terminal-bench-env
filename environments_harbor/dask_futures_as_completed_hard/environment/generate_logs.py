#!/usr/bin/env python3

import random
import time
import json
import itertools

def generate_log_entries(num_entries=1000, num_sources=5):
    """
    Generator that yields log entry dictionaries with varying processing times.
    """
    # Set seed for deterministic testing
    random.seed(42)
    
    # Define processing time ranges for each source
    source_times = {
        'source_0': (0.1, 0.3),   # fast
        'source_1': (0.5, 1.0),   # medium
        'source_2': (1.0, 2.0),   # slow
        'source_3': (0.2, 0.5),   # fast-medium
        'source_4': (2.0, 3.0),   # very slow
    }
    
    severities = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
    
    for i in range(num_entries):
        source = f'source_{i % num_sources}'
        min_time, max_time = source_times[source]
        
        log_entry = {
            'id': i,
            'source': source,
            'timestamp': time.time(),
            'message': f'Log entry {i} from {source}',
            'severity': random.choice(severities),
            'processing_time': random.uniform(min_time, max_time)
        }
        
        yield log_entry

def get_all_logs(num_entries=1000, num_sources=5):
    """
    Returns a list of all generated log entries.
    """
    return list(generate_log_entries(num_entries, num_sources))

if __name__ == '__main__':
    # Test the generator
    print("Testing log generator...")
    print("\nFirst 10 log entries:")
    
    for i, entry in enumerate(itertools.islice(generate_log_entries(), 10)):
        print(f"\nEntry {i}:")
        print(json.dumps(entry, indent=2))
    
    # Test get_all_logs
    all_logs = get_all_logs(20, 5)
    print(f"\n\nGenerated {len(all_logs)} total logs")
    print(f"Sources: {set(log['source'] for log in all_logs)}")
    print(f"Severities: {set(log['severity'] for log in all_logs)}")
    
    # Show processing time distribution
    print("\nProcessing time ranges by source:")
    for source in ['source_0', 'source_1', 'source_2', 'source_3', 'source_4']:
        source_logs = [log for log in all_logs if log['source'] == source]
        if source_logs:
            times = [log['processing_time'] for log in source_logs]
            print(f"{source}: {min(times):.3f}s - {max(times):.3f}s")