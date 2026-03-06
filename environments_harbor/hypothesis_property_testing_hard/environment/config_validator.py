#!/usr/bin/env python3

class ConfigValidator:
    """Validates game server configuration dictionaries."""
    
    def __init__(self):
        self.valid_map_sizes = ["small", "medium", "large"]
    
    def validate(self, config_dict):
        """
        Validates a game server configuration.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config_dict, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Check required fields
        required_fields = ["max_players", "min_players", "map_size", "resource_multiplier"]
        for field in required_fields:
            if field not in config_dict:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate max_players (BUG: should be >= 2, but using > 2)
        max_players = config_dict["max_players"]
        if not isinstance(max_players, int):
            raise ValueError("max_players must be an integer")
        if max_players < 2 or max_players > 100:
            raise ValueError("max_players must be between 2 and 100 (inclusive)")
        
        # Validate min_players
        min_players = config_dict["min_players"]
        if not isinstance(min_players, int):
            raise ValueError("min_players must be an integer")
        if min_players > max_players:
            raise ValueError("min_players must be less than or equal to max_players")
        
        # Validate map_size
        map_size = config_dict["map_size"]
        if not isinstance(map_size, str):
            raise ValueError("map_size must be a string")
        if map_size not in self.valid_map_sizes:
            raise ValueError(f"map_size must be one of: {', '.join(self.valid_map_sizes)}")
        
        # Validate resource_multiplier (BUG: missing upper bound check)
        resource_multiplier = config_dict["resource_multiplier"]
        if not isinstance(resource_multiplier, (int, float)):
            raise ValueError("resource_multiplier must be a number")
        if resource_multiplier < 0.1:
            raise ValueError("resource_multiplier must be between 0.1 and 10.0")
        
        # Validate teams (optional)
        if "teams" in config_dict:
            teams = config_dict["teams"]
            if not isinstance(teams, list):
                raise ValueError("teams must be a list")
            
            # BUG: should check length between 2 and 4, but checking < 2 or >= 4 (off-by-one)
            if len(teams) < 2 or len(teams) >= 4:
                raise ValueError("teams must have length between 2 and 4")
            
            # Validate team min_players sum
            total_team_min_players = 0
            for i, team in enumerate(teams):
                if not isinstance(team, dict):
                    raise ValueError(f"Team {i} must be a dictionary")
                if "min_players" not in team:
                    raise ValueError(f"Team {i} missing min_players")
                team_min = team["min_players"]
                if not isinstance(team_min, int) or team_min < 0:
                    raise ValueError(f"Team {i} min_players must be a non-negative integer")
                total_team_min_players += team_min
            
            # BUG: should check total <= max_players, but checking < max_players
            if total_team_min_players > max_players:
                raise ValueError("Total team min_players cannot exceed max_players")
        
        return True