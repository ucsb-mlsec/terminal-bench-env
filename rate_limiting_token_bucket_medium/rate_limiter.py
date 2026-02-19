#!/usr/bin/env python3
"""
Rate Limiter Module - Token Bucket Implementation
"""

import time
from typing import Dict


class RateLimiter:
    """
    A token bucket rate limiter implementation.
    
    This class implements a token bucket algorithm to limit the rate of requests
    from different clients. Each client has their own bucket that refills at a
    constant rate.
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize the rate limiter.
        
        Args:
            capacity: Maximum number of tokens a bucket can hold
            refill_rate: Number of tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        # Dictionary to store bucket state for each client
        # Each entry: client_id -> {'tokens': float, 'last_refill': float}
        self.buckets: Dict[str, Dict[str, float]] = {}
    
    def _refill_tokens(self, client_id: str) -> None:
        """
        Refill tokens for a client based on elapsed time.
        
        Args:
            client_id: The client identifier
        """
        current_time = time.time()
        
        if client_id not in self.buckets:
            # New client, initialize with full capacity
            self.buckets[client_id] = {
                'tokens': float(self.capacity),
                'last_refill': current_time
            }
            return
        
        bucket = self.buckets[client_id]
        time_elapsed = current_time - bucket['last_refill']
        
        # Calculate tokens to add based on elapsed time
        tokens_to_add = time_elapsed * self.refill_rate
        
        # Update token count (capped at capacity)
        bucket['tokens'] = min(self.capacity, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = current_time
    
    def allow_request(self, client_id: str) -> bool:
        """
        Check if a request should be allowed for a client.
        
        Args:
            client_id: The client identifier
            
        Returns:
            True if request is allowed (token available), False otherwise
        """
        # Refill tokens based on time elapsed
        self._refill_tokens(client_id)
        
        bucket = self.buckets[client_id]
        
        # Check if at least 1 token is available
        if bucket['tokens'] >= 1.0:
            # Consume 1 token
            bucket['tokens'] -= 1.0
            return True
        else:
            # No tokens available
            return False
    
    def get_tokens(self, client_id: str) -> float:
        """
        Get the current token count for a client.
        
        Args:
            client_id: The client identifier
            
        Returns:
            Current number of tokens available for the client
        """
        # Refill tokens before returning count
        self._refill_tokens(client_id)
        
        if client_id not in self.buckets:
            return float(self.capacity)
        
        return self.buckets[client_id]['tokens']