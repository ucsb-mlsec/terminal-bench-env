#!/usr/bin/env python3
# Mystery Writer Process

import time
import datetime
import os
import random

# Non-obvious location for data output
OUTPUT_DIR = "/var/tmp/.cache_worker"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "proc_data_8f3a2c.dat")

def setup_output_location():
    """Create the output directory if it doesn't exist"""
    try:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, mode=0o755)
        return True
    except Exception as e:
        print(f"Error creating directory: {e}")
        return False

def write_data_entry():
    """Write a timestamped data entry to the file"""
    try:
        timestamp = datetime.datetime.now().isoformat()
        random_value = random.randint(1000, 9999)
        status = random.choice(['ACTIVE', 'PROCESSING', 'READY', 'IDLE'])
        
        data_line = f"{timestamp},{random_value},{status}\n"
        
        with open(OUTPUT_FILE, 'a') as f:
            f.write(data_line)
            f.flush()
        
        return True
    except Exception as e:
        print(f"Error writing data: {e}")
        return False

def main():
    """Main loop - continuously write data"""
    if not setup_output_location():
        print("Failed to setup output location")
        return
    
    print(f"Mystery writer started. PID: {os.getpid()}")
    
    while True:
        try:
            write_data_entry()
            time.sleep(2.5)
        except KeyboardInterrupt:
            print("\nShutting down mystery writer...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()