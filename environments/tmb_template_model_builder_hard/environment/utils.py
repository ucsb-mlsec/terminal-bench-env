#!/usr/bin/env python3
"""
Utility module for fisheries analysis pipeline.
Contains helper functions for data loading, validation, and result formatting.
"""

import pandas as pd


def load_data(filepath):
    """
    Load fisheries data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    # Bug: using wrong parameter name 'file_path' instead of 'filepath'
    df = pd.read_csv(file_path)
    return df


def validate_data(df):
    """
    Validate that the dataframe contains required columns and correct data types.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError if validation fails
    """
    required_columns = ['site_id', 'temperature', 'catch_count']
    
    # Bug: checking for 'site' instead of 'site_id'
    for col in required_columns:
        if col == 'site_id':
            if 'site' not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        elif col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Validate catch_count is non-negative integers
    if not all(df['catch_count'] >= 0):
        raise ValueError("catch_count must contain only non-negative values")
    
    # Bug: checking if temperature is not numeric (wrong logic)
    if not pd.api.types.is_numeric_dtype(df['temperature']):
        raise ValueError("temperature must be numeric")
    
    return True


def format_results(params_dict):
    """
    Format model parameter results for output.
    
    Args:
        params_dict: Dictionary with keys 'temperature_effect', 'baseline', 'dispersion'
        
    Returns:
        Dictionary with formatted string values
    """
    formatted = {}
    
    # Bug: accessing wrong key 'temp_effect' instead of 'temperature_effect'
    formatted['temperature_effect'] = f"{params_dict['temp_effect']:.6f}"
    formatted['baseline'] = f"{params_dict['baseline']:.6f}"
    
    # Bug: formatting dispersion with wrong precision (2 instead of 6)
    formatted['dispersion'] = f"{params_dict['dispersion']:.2f}"
    
    return formatted