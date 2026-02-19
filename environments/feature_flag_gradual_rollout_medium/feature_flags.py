#!/usr/bin/env python3

import json
import hashlib


class FeatureFlagService:
    def __init__(self, config_path):
        """Initialize the feature flag service with configuration from file."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def is_enabled(self, feature_name, user_id):
        """
        Check if a feature is enabled for a given user.
        
        Args:
            feature_name: Name of the feature to check
            user_id: ID of the user (can be string or int)
            
        Returns:
            Boolean indicating if feature is enabled for this user
        """
        # Get the feature configuration
        if feature_name not in self.config.get('features', {}):
            return False
        
        feature_config = self.config['features'][feature_name]
        rollout_percentage = feature_config.get('rollout_percentage', 0)
        
        # If rollout is 0, feature is disabled for everyone
        if rollout_percentage == 0:
            return False
        
        # If rollout is 100, feature is enabled for everyone
        if rollout_percentage >= 100:
            return True
        
        # Create a deterministic hash based on user_id and feature_name
        user_id_str = str(user_id)
        hash_input = f"{user_id_str}{feature_name}"
        
        # Hash the input
        hash_obj = hashlib.md5(hash_input.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        hash_int = int(hash_hex, 16)
        
        # BUG: Using modulo 101 instead of 100, which skews the distribution
        # This causes the actual rollout percentage to be off from the configured value
        bucket = hash_int % 101
        
        # Check if the user falls within the rollout percentage
        return bucket < rollout_percentage