#!/usr/bin/env python3

class LWWRegister:
    """Last-Write-Wins Register CRDT implementation"""
    
    def __init__(self):
        """Initialize an empty LWW register"""
        self.value = None
        self.timestamp = 0
    
    def update(self, value, timestamp):
        """Update the register with a new value and timestamp"""
        if timestamp > self.timestamp:
            self.value = value
            self.timestamp = timestamp
        elif timestamp == self.timestamp:
            # Tie-breaking: use lexicographic comparison of values
            if value is not None and self.value is not None:
                if value > self.value:
                    self.value = value
            elif value is not None:
                self.value = value
    
    def merge(self, other_register):
        """
        Merge state from another LWWRegister instance.
        BUG: Using < instead of > for timestamp comparison
        """
        other_state = other_register.get_state()
        other_timestamp = other_state['timestamp']
        other_value = other_state['value']
        
        # BUG: This comparison is backwards!
        # Should accept the other value if other_timestamp > self.timestamp
        # But this code does the opposite
        if other_timestamp < self.timestamp:
            self.value = other_value
            self.timestamp = other_timestamp
        elif other_timestamp == self.timestamp:
            # Tie-breaking: use lexicographic comparison
            if other_value is not None and self.value is not None:
                if other_value > self.value:
                    self.value = other_value
            elif other_value is not None:
                self.value = other_value
    
    def get_value(self):
        """Return the current value"""
        return self.value
    
    def get_state(self):
        """Return the current state (value and timestamp)"""
        return {
            'value': self.value,
            'timestamp': self.timestamp
        }