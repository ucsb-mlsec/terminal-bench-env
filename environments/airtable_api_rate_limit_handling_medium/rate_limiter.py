#!/usr/bin/env python3

import time
from collections import deque


class RateLimitError(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, wait_time):
        self.wait_time = wait_time
        super().__init__(f"Rate limit exceeded. Please wait {wait_time:.2f} seconds before retrying.")


class RateLimitedProcessor:
    """Simulates Airtable's rate limiting of 5 requests per second."""
    
    def __init__(self, max_calls=5, time_window=1.0):
        """
        Initialize the rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed per time window (default: 5)
            time_window: Time window in seconds (default: 1.0)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.call_timestamps = deque()
    
    def process_record(self, record_id):
        """
        Process a record with rate limiting.
        
        Args:
            record_id: The ID of the record to process
            
        Returns:
            True if the record was processed successfully
            
        Raises:
            RateLimitError: If the rate limit is exceeded
        """
        current_time = time.time()
        
        # Remove timestamps older than the time window
        while self.call_timestamps and current_time - self.call_timestamps[0] >= self.time_window:
            self.call_timestamps.popleft()
        
        # Check if we've exceeded the rate limit
        if len(self.call_timestamps) >= self.max_calls:
            # Calculate how long to wait until the oldest call expires
            oldest_call = self.call_timestamps[0]
            wait_time = self.time_window - (current_time - oldest_call)
            raise RateLimitError(wait_time)
        
        # Record this call and process the record
        self.call_timestamps.append(current_time)
        return True