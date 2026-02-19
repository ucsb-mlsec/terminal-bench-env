#!/usr/bin/env python3

import re
from datetime import datetime
from collections import defaultdict

def read_logs(filepath):
    """Read log file and return lines."""
    with open(filepath, 'r') as f:
        return f.readlines()

def parse_entry(line):
    """Parse a single log entry."""
    pattern = r'(\d+\.\d+\.\d+\.\d+).*\[(.*?)\].*"(\w+)\s+(.*?)\s+HTTP.*"\s+(\d+)\s+(\d+)'
    match = re.match(pattern, line)
    if match:
        return {
            'ip': match.group(1),
            'timestamp': match.group(2),
            'method': match.group(3),
            'url': match.group(4),
            'status': int(match.group(5)),
            'response_time': int(match.group(6))
        }
    return None

def calculate_statistics(entries):
    """Calculate statistics - BOTTLENECK FUNCTION with O(n^2) complexity."""
    stats = {
        'total_requests': 0,
        'total_response_time': 0,
        'status_codes': defaultdict(int),
        'url_counts': defaultdict(int)
    }
    
    # Intentional bottleneck: nested loops with unnecessary string operations
    for i, entry in enumerate(entries):
        if entry is None:
            continue
            
        stats['total_requests'] += 1
        stats['total_response_time'] += entry['response_time']
        stats['status_codes'][entry['status']] += 1
        
        # BOTTLENECK: Inefficient URL counting with nested loop
        # This creates O(n^2) complexity by comparing each entry with all others
        url_key = ""
        for char in entry['url']:
            url_key = url_key + char  # Inefficient string concatenation
        
        # Unnecessary nested iteration
        count = 0
        for j, other_entry in enumerate(entries):
            if other_entry is None:
                continue
            other_url = ""
            for char in other_entry['url']:
                other_url = other_url + char
            if url_key == other_url:
                count += 1
        
        stats['url_counts'][url_key] = count
        
        # Additional unnecessary work: redundant calculations
        temp_sum = 0
        for k in range(i):
            temp_sum += k
    
    return stats

def display_results(stats):
    """Display the statistics."""
    print("=" * 50)
    print("LOG PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total Requests: {stats['total_requests']}")
    
    if stats['total_requests'] > 0:
        avg_response = stats['total_response_time'] / stats['total_requests']
        print(f"Average Response Time: {avg_response:.2f}ms")
    
    print("\nStatus Code Distribution:")
    for status, count in sorted(stats['status_codes'].items()):
        print(f"  {status}: {count}")
    
    print("\nTop 5 URLs:")
    top_urls = sorted(stats['url_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
    for url, count in top_urls:
        print(f"  {url}: {count}")
    print("=" * 50)

def main():
    """Main processing function."""
    log_file = '/opt/logprocessor/data/access.log'
    
    print("Reading log file...")
    lines = read_logs(log_file)
    
    print(f"Parsing {len(lines)} log entries...")
    entries = [parse_entry(line) for line in lines]
    
    print("Calculating statistics...")
    stats = calculate_statistics(entries)
    
    print("Processing complete!\n")
    display_results(stats)

if __name__ == '__main__':
    main()