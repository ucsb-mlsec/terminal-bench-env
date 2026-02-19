#!/usr/bin/env python3

import copy
from collections import defaultdict

class ConfigVersionControl:
    """
    Configuration version control system using delta storage.
    
    Instead of storing complete copies of configuration for each version,
    this system stores only the changes (deltas) between versions.
    This dramatically reduces memory usage and improves performance.
    """
    
    def __init__(self):
        """
        Initialize the configuration version control system.
        
        Data structures:
        - current_config: Working copy of the current configuration (not yet committed)
        - deltas: List of changes for each version. Each delta is a dict of {key: value}
        - base_config: The initial configuration state (version 0)
        """
        self.current_config = {}
        self.deltas = []  # Each entry is a dict of changes for that version
        self.base_config = {}  # Starting point
        self.last_committed = {}  # Cache of last committed state
    
    def set(self, key, value):
        """
        Update a configuration value in the current working state.
        
        Args:
            key: Configuration key (string)
            value: Configuration value (string)
        """
        self.current_config[key] = value
    
    def commit(self):
        """
        Create a new version snapshot by storing only the changes (delta).
        
        This is the optimized implementation that stores only what changed
        rather than copying the entire configuration.
        """
        if not self.deltas:
            # First commit - store as base configuration
            self.base_config = copy.deepcopy(self.current_config)
            self.last_committed = copy.deepcopy(self.current_config)
            self.deltas.append({})  # Empty delta for base version
        else:
            # Store only the changes from last committed state
            delta = {}
            
            # Find keys that changed or are new
            for key, value in self.current_config.items():
                if key not in self.last_committed or self.last_committed[key] != value:
                    delta[key] = value
            
            # Find keys that were deleted
            for key in self.last_committed:
                if key not in self.current_config:
                    delta[key] = None  # Use None to mark deletions
            
            self.deltas.append(delta)
            self.last_committed = copy.deepcopy(self.current_config)
    
    def get(self, key, version=None):
        """
        Retrieve a configuration value at a specific version.
        
        Args:
            key: Configuration key to retrieve
            version: Version number (None = latest)
            
        Returns:
            Configuration value or None if key doesn't exist
        """
        if version is None:
            version = len(self.deltas) - 1
        
        if version < 0 or version >= len(self.deltas):
            return None
        
        # Start from base and apply deltas up to requested version
        config = self._reconstruct_version(version)
        return config.get(key)
    
    def get_all(self, version=None):
        """
        Get all configuration settings at the specified version.
        
        Args:
            version: Version number (None = latest)
            
        Returns:
            Dictionary of all configuration settings at that version
        """
        if version is None:
            version = len(self.deltas) - 1
        
        if version < 0 or version >= len(self.deltas):
            return {}
        
        return self._reconstruct_version(version)
    
    def _reconstruct_version(self, version):
        """
        Reconstruct the complete configuration state at a given version.
        
        This method applies all deltas from base up to the requested version.
        
        Args:
            version: Version number to reconstruct
            
        Returns:
            Dictionary representing the full configuration at that version
        """
        # Start with base configuration
        config = copy.deepcopy(self.base_config)
        
        # Apply each delta up to and including the requested version
        for i in range(1, version + 1):
            if i < len(self.deltas):
                delta = self.deltas[i]
                for key, value in delta.items():
                    if value is None:
                        # None means the key was deleted
                        config.pop(key, None)
                    else:
                        config[key] = value
        
        return config