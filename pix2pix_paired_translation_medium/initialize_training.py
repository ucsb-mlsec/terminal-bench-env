#!/usr/bin/env python3

import yaml
import sys
import os
from model_architecture import Pix2PixModel


def load_config(filepath):
    """
    Load YAML configuration file.
    
    Args:
        filepath: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_and_initialize(config):
    """
    Validate configuration and initialize Pix2PixModel.
    
    Args:
        config: Dictionary containing model configuration
        
    Returns:
        Tuple of (success: bool, error_message: str or None, model: Pix2PixModel or None)
    """
    try:
        model = Pix2PixModel(config)
        return (True, None, model)
    except Exception as e:
        error_message = f"{type(e).__name__}: {str(e)}"
        return (False, error_message, None)


if __name__ == "__main__":
    print("=" * 60)
    print("Pix2Pix Model Initialization")
    print("=" * 60)
    
    config_path = 'config/model_config.yaml'
    
    try:
        # Load configuration
        print(f"\nLoading configuration from: {config_path}")
        config = load_config(config_path)
        print("✓ Configuration file loaded successfully")
        
        # Validate and initialize model
        print("\nAttempting to initialize Pix2PixModel...")
        success, error_message, model = validate_and_initialize(config)
        
        # Generate report
        report_path = '/tmp/config_fix_report.txt'
        
        if success:
            print("✓ Model initialized successfully!")
            print(f"\nModel Details:")
            print(f"  - Generator architecture: {config.get('generator', {}).get('architecture', 'N/A')}")
            print(f"  - Discriminator architecture: {config.get('discriminator', {}).get('architecture', 'N/A')}")
            print(f"  - Input channels: {config.get('input_channels', 'N/A')}")
            print(f"  - Output channels: {config.get('output_channels', 'N/A')}")
            
            with open(report_path, 'w') as f:
                f.write("INITIALIZATION_STATUS: SUCCESS\n")
                f.write("ISSUES_FOUND: 0\n")
                f.write("PRIMARY_FIX: configuration successfully validated\n")
            
            print(f"\n✓ Report saved to: {report_path}")
        else:
            print("✗ Model initialization failed!")
            print(f"\nError Details:")
            print(f"  {error_message}")
            
            with open(report_path, 'w') as f:
                f.write("INITIALIZATION_STATUS: FAILED\n")
                f.write("ISSUES_FOUND: 0\n")
                f.write("PRIMARY_FIX: \n")
            
            print(f"\n✗ Report saved to: {report_path}")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {str(e)}")
        sys.exit(1)