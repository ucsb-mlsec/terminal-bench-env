#!/usr/bin/env python3

import os
import sys

def monitor_processes(pid_file):
    """
    Attempt to monitor process status by reading PIDs from a file
    and using waitpid to check their status.
    
    NOTE: I'm confused why this keeps failing with ECHILD errors!
    The documentation says waitpid() should wait for process state changes...
    """
    
    # Read PIDs from the file
    try:
        with open(pid_file, 'r') as f:
            pids = [int(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Could not find {pid_file}")
        return
    except ValueError:
        print("Error: Invalid PID format in file")
        return
    
    print(f"Attempting to monitor {len(pids)} processes...")
    
    # Try to monitor each process
    for pid in pids:
        print(f"\nMonitoring PID {pid}...")
        try:
            # Wait for the process to change state
            # Why doesn't this work?? The PID exists in the system!
            result = os.waitpid(pid, os.WNOHANG)
            print(f"Success! Process {pid} status: {result}")
        except OSError as e:
            if e.errno == 10:  # ECHILD error number
                print(f"ERROR: Got ECHILD for PID {pid}")
                print("Why is this failing? The process exists!")
                print("I checked with 'ps' and the process is running...")
            else:
                print(f"Unexpected error for PID {pid}: {e}")
        except Exception as e:
            print(f"Unexpected exception for PID {pid}: {e}")
    
    # Maybe try a general wait() call?
    print("\n--- Trying general wait() call ---")
    try:
        # This should work for ANY process, right?
        result = os.wait()
        print(f"wait() returned: {result}")
    except OSError as e:
        print(f"wait() also failed with: {e}")
        print("I don't understand why wait/waitpid won't work here!")

if __name__ == "__main__":
    pid_file = "sample_pids.txt"
    if len(sys.argv) > 1:
        pid_file = sys.argv[1]
    
    monitor_processes(pid_file)