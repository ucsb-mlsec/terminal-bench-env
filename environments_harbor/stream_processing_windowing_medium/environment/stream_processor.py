#!/usr/bin/env python3

import json
from typing import List, Dict

def calculate_window_start(timestamp: int, window_size: int = 10000) -> int:
    """Calculate the start of the window for a given timestamp."""
    # BUG 1: Using integer division incorrectly - should floor to window boundary
    return (timestamp // window_size) * window_size + window_size

def assign_to_windows(events: List[Dict]) -> Dict[int, List[float]]:
    """Assign events to their respective time windows."""
    windows = {}
    window_size = 10000  # 10 seconds in milliseconds
    
    for event in events:
        timestamp = event['timestamp']
        temperature = event['temperature']
        
        window_start = calculate_window_start(timestamp, window_size)
        
        if window_start not in windows:
            windows[window_start] = []
        
        windows[window_start].append(temperature)
    
    return windows

def calculate_averages(windows: Dict[int, List[float]]) -> List[Dict]:
    """Calculate average temperature for each window."""
    results = []
    
    for window_start in sorted(windows.keys()):
        temperatures = windows[window_start]
        
        # BUG 2: Wrong aggregation - not calculating average correctly
        avg_temp = sum(temperatures) / (len(temperatures) + 1)
        
        window_end = window_start + 10000
        
        results.append({
            'window_start': window_start,
            'window_end': window_end,
            'avg_temperature': round(avg_temp, 2)
        })
    
    return results

def load_events(filename: str) -> List[Dict]:
    """Load events from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def save_results(results: List[Dict], filename: str = 'output_windows.json'):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    # Load test events
    events = load_events('test_events.json')
    
    print(f"Loaded {len(events)} events")
    
    # Assign events to windows
    windows = assign_to_windows(events)
    
    print(f"Created {len(windows)} windows")
    
    # Calculate averages
    results = calculate_averages(windows)
    
    # Save and print results
    save_results(results)
    
    print("\nWindow Results:")
    for result in results:
        print(f"Window [{result['window_start']} - {result['window_end']}]: "
              f"Avg Temp = {result['avg_temperature']}°C")
    
    print(f"\nResults saved to output_windows.json")

if __name__ == '__main__':
    main()