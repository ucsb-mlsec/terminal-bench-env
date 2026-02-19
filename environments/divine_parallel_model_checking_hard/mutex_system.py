#!/usr/bin/env python3

import time

class SharedCounter:
    """
    A shared counter with flawed mutex implementation.
    This class attempts to provide thread-safe access to a counter
    using a simple flag-based locking mechanism.
    WARNING: This implementation has race conditions!
    """
    
    def __init__(self):
        self.counter = 0
        self.lock = False  # Simple flag-based lock
    
    def acquire(self):
        """
        Attempt to acquire the lock.
        This implementation is NOT atomic and has race conditions!
        """
        # Wait until lock appears to be free
        while self.lock:
            time.sleep(0.00001)  # Small delay
        
        # Set the lock - BUT this is not atomic with the check above!
        # Multiple threads can pass the while loop before any sets lock=True
        self.lock = True
    
    def release(self):
        """Release the lock."""
        self.lock = False
    
    def increment(self):
        """
        Increment the counter with attempted mutual exclusion.
        This should be thread-safe, but the lock mechanism is flawed.
        """
        self.acquire()
        
        # Critical section - read, modify, write
        current = self.counter
        time.sleep(0.00001)  # Simulate some processing time
        self.counter = current + 1
        
        self.release()
    
    def get_value(self):
        """Get the current counter value."""
        return self.counter
    
    def reset(self):
        """Reset the counter to zero."""
        self.counter = 0
        self.lock = False
