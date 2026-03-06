#!/usr/bin/env python3

import time
import random
from typing import Optional, Dict, Tuple

class MockStorageNode:
    """Simulates an independent key-value store with realistic failure scenarios."""
    
    def __init__(self, node_id: int):
        """
        Initialize a storage node.
        
        Args:
            node_id: Identifier for this node
        """
        self.node_id = node_id
        self.storage: Dict[str, Tuple[str, float]] = {}
        # Simulate clock drift: each node has a slight time offset (-50ms to +50ms)
        self.clock_drift = random.uniform(-0.05, 0.05)
        
    def _get_current_time(self) -> float:
        """Get current time with simulated clock drift."""
        return time.time() + self.clock_drift
    
    def _simulate_network_delay(self):
        """Simulate network latency (0-100ms)."""
        time.sleep(random.uniform(0, 0.1))
    
    def _simulate_failure(self):
        """Simulate occasional node unavailability (~7% probability)."""
        if random.random() < 0.07:
            raise Exception(f"Node {self.node_id} temporarily unavailable")
    
    def _clean_expired(self, key: str):
        """Remove key if it has expired."""
        if key in self.storage:
            _, expiration = self.storage[key]
            if self._get_current_time() >= expiration:
                del self.storage[key]
    
    def set_if_not_exists(self, key: str, value: str, ttl_seconds: int) -> bool:
        """
        Atomically set a key if it doesn't exist.
        
        Args:
            key: The key to set
            value: The value to store
            ttl_seconds: Time-to-live in seconds
            
        Returns:
            True if key was set successfully, False if key already exists
            
        Raises:
            Exception: If node is temporarily unavailable
        """
        self._simulate_network_delay()
        self._simulate_failure()
        
        # Clean up expired key if it exists
        self._clean_expired(key)
        
        # Check if key exists and is not expired
        if key in self.storage:
            return False
        
        # Set the key with expiration time
        expiration_time = self._get_current_time() + ttl_seconds
        self.storage[key] = (value, expiration_time)
        return True
    
    def delete_if_matches(self, key: str, value: str) -> bool:
        """
        Delete a key only if its value matches.
        
        Args:
            key: The key to delete
            value: The value that must match
            
        Returns:
            True if key was deleted, False if key doesn't exist or value doesn't match
            
        Raises:
            Exception: If node is temporarily unavailable
        """
        self._simulate_network_delay()
        self._simulate_failure()
        
        # Clean up expired key if it exists
        self._clean_expired(key)
        
        # Check if key exists and value matches
        if key not in self.storage:
            return False
        
        stored_value, _ = self.storage[key]
        if stored_value != value:
            return False
        
        # Delete the key
        del self.storage[key]
        return True
    
    def get(self, key: str) -> Optional[str]:
        """
        Retrieve the value for a key.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The value if found and not expired, None otherwise
            
        Raises:
            Exception: If node is temporarily unavailable
        """
        self._simulate_network_delay()
        self._simulate_failure()
        
        # Clean up expired key if it exists
        self._clean_expired(key)
        
        # Return value if key exists
        if key in self.storage:
            value, _ = self.storage[key]
            return value
        
        return None