#!/usr/bin/env python3

import json
import os
from datetime import datetime

class SecretRotationStateMachine:
    """
    Broken state machine for managing database credential rotation.
    Contains multiple bugs in state transition logic and validation.
    """
    
    VALID_STATES = ['CREATE', 'SET', 'TEST', 'FINISH']
    
    def __init__(self, state_file_path):
        self.state_file_path = state_file_path
        self.state = self._load_state()
    
    def _load_state(self):
        """Load state from JSON file"""
        if not os.path.exists(self.state_file_path):
            raise FileNotFoundError(f"State file not found: {self.state_file_path}")
        
        with open(self.state_file_path, 'r') as f:
            return json.load(f)
    
    def save_state(self):
        """Save current state to JSON file"""
        with open(self.state_file_path, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_current_state(self):
        """Returns current state name"""
        return self.state.get('current_state', 'UNKNOWN')
    
    def transition(self, target_state):
        """
        Attempt to transition to target state.
        BUG: Allows invalid transitions and doesn't check preconditions!
        """
        current = self.state.get('current_state')
        
        # BUG 1: No validation that target_state is valid
        # BUG 2: Allows skipping states - no check for proper sequence
        if target_state in self.VALID_STATES:
            self.state['current_state'] = target_state
            # BUG 3: last_updated timestamp is NOT updated
            
            # BUG 4: Metadata flags are NOT updated based on state
            # Should set credentials_generated=True in CREATE
            # Should set credentials_applied=True in SET
            # Should set credentials_tested=True in TEST
            # Should set rotation_complete=True in FINISH
            
            self.save_state()
            return True
        
        return False
    
    def validate_state(self):
        """
        Check if current state is consistent.
        BUG: Always returns True, no actual validation!
        """
        # BUG 5: No validation of preconditions
        # Should check:
        # - If state is SET, credentials_generated should be True
        # - If state is TEST, credentials_applied should be True
        # - If state is FINISH, credentials_tested should be True
        
        # BUG 6: Missing checks for required fields
        current = self.state.get('current_state')
        if current:
            return True  # BUG: Always returns True!
        
        return True

    def _validate_transition(self, target_state):
        """
        Validate if transition to target state is allowed.
        BUG: Not actually called by transition() method!
        """
        current = self.state.get('current_state')
        
        # This would be the correct logic but it's never used!
        state_order = {'CREATE': 0, 'SET': 1, 'TEST': 2, 'FINISH': 3}
        
        if current not in state_order or target_state not in state_order:
            return False
        
        # Should only allow moving to next state in sequence
        if state_order[target_state] != state_order[current] + 1:
            return False
        
        return True
    
    def _check_preconditions(self, target_state):
        """
        Check preconditions for transitioning to target state.
        BUG: Not actually called by transition() method!
        """
        # This would check the right preconditions but it's not used!
        if target_state == 'SET' and not self.state.get('credentials_generated'):
            return False
        if target_state == 'TEST' and not self.state.get('credentials_applied'):
            return False
        if target_state == 'FINISH' and not self.state.get('credentials_tested'):
            return False
        
        return True

if __name__ == "__main__":
    # Simple test harness
    machine = SecretRotationStateMachine('/workspace/secret_state.json')
    print(f"Current state: {machine.get_current_state()}")
    print(f"State valid: {machine.validate_state()}")