#!/usr/bin/env python3
"""
Parsl Configuration Validator

This script validates Parsl configuration files, specifically checking
timeout parameters and worker configuration to ensure proper initialization.
"""

import sys
import json
import importlib.util


def validate_config(config_path):
    """
    Validate a Parsl configuration file.
    
    Args:
        config_path: Path to the Python configuration file
        
    Returns:
        Dictionary containing validation results with keys:
        - valid: Boolean indicating if config is valid
        - worker_init_timeout: The timeout value found
        - max_workers: Number of workers configured
        - errors: List of error messages
    """
    result = {
        'valid': False,
        'worker_init_timeout': None,
        'max_workers': None,
        'errors': []
    }
    
    try:
        # Dynamically import the configuration module from file path
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        if spec is None or spec.loader is None:
            result['errors'].append(f"Cannot load module from {config_path}")
            return result
            
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Extract the config object from the module
        if not hasattr(config_module, 'config'):
            result['errors'].append("Module does not contain 'config' object")
            return result
            
        config = config_module.config
        
        # Check if config has executors
        if not hasattr(config, 'executors') or not config.executors:
            result['errors'].append("Config has no executors defined")
            return result
        
        # Validate each executor
        for idx, executor in enumerate(config.executors):
            # Check for worker_init_timeout attribute
            if not hasattr(executor, 'worker_init_timeout'):
                result['errors'].append(f"Executor {idx} missing worker_init_timeout")
                continue
                
            # Check for max_workers attribute
            if not hasattr(executor, 'max_workers'):
                result['errors'].append(f"Executor {idx} missing max_workers")
                continue
            
            # Validate worker_init_timeout is a number
            timeout_val = executor.worker_init_timeout
            if not isinstance(timeout_val, (int, float)):
                result['errors'].append(f"Executor {idx} worker_init_timeout is not a number")
                continue
                
            # Store the values from first valid executor
            if result['worker_init_timeout'] is None:
                result['worker_init_timeout'] = timeout_val
                result['max_workers'] = executor.max_workers
        
        # Configuration is valid if we found timeout and workers with no errors
        if result['worker_init_timeout'] is not None and result['max_workers'] is not None:
            result['valid'] = len(result['errors']) == 0
            
    except Exception as e:
        result['errors'].append(f"Exception during validation: {str(e)}")
    
    return result


def main():
    """
    Main entry point for the validation script.
    
    Reads configuration file path from command line arguments,
    validates the configuration, and prints results in JSON format.
    """
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: validate_config.py <config_file_path>'}))
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Validate the configuration file
    validation_result = validate_config(config_path)
    
    # Print results in JSON format
    print(json.dumps(validation_result, indent=2))
    
    # Exit with appropriate code
    if validation_result['valid']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()