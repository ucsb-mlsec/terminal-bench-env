#!/usr/bin/env python3

import sys
from datetime import datetime

def generate_logs():
    """Generate exactly 100 log lines"""
    log_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
    
    for i in range(100):
        timestamp = datetime.now().isoformat()
        level = log_levels[i % len(log_levels)]
        message = f"Log entry {i+1}"
        log_line = f"{timestamp} [{level}] {message}"
        
        print(log_line)
        sys.stdout.flush()
    
    # Cleanup operations
    pass

if __name__ == "__main__":
    generate_logs()