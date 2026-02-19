#!/usr/bin/env python3

import sys
import os
from datetime import datetime
import json

def analyze_directory(path):
    """
    Analyze a directory and return statistics about its contents.
    
    Args:
        path: Directory path to analyze
        
    Returns:
        Dictionary containing file statistics
    """
    stats = {
        'total_files': 0,
        'total_dirs': 0,
        'total_size': 0,
        'file_types': {}
    }
    
    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist")
        return stats
    
    for root, dirs, files in os.walk(path):
        stats['total_dirs'] += len(dirs)
        stats['total_files'] += len(files)
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                stats['total_size'] += size
                
                ext = os.path.splitext(file)[1] or 'no_extension'
                stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
            except OSError:
                continue
    
    return stats

def format_size(bytes_size):
    """
    Convert bytes to human-readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted string with appropriate unit
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def print_report(stats, path):
    """
    Print a formatted report of directory statistics.
    
    Args:
        stats: Statistics dictionary
        path: Path that was analyzed
    """
    print(f"\n{'='*60}")
    print(f"Directory Analysis Report")
    print(f"{'='*60}")
    print(f"Path: {path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTotal Directories: {stats['total_dirs']}")
    print(f"Total Files: {stats['total_files']}")
    print(f"Total Size: {format_size(stats['total_size'])}")
    print(f"\nFile Types Distribution:")
    for ext, count in sorted(stats['file_types'].items()):
        print(f"  {ext}: {count} files")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    target_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    print(f"Analyzing directory: {target_path}")
    statistics = analyze_directory(target_path)
    print_report(statistics, target_path)
    
    output_file = 'analysis_result.json'
    with open(output_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"Results saved to {output_file}")