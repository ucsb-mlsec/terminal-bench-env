#!/usr/bin/env python3

import os
import sys

def read_config(filepath):
    """Read key=value configuration file and return dictionary"""
    config = {}
    if not os.path.exists(filepath):
        return config
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Parse key=value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config

def validate_extension(extension_name):
    """Check if extension matches the correct identifier"""
    valid_extensions = ['GL_NV_conservative_raster', 'GL_INTEL_conservative_rasterization']
    if extension_name in valid_extensions:
        return True, "Extension valid"
    return False, f"Invalid extension: {extension_name}. Expected one of {valid_extensions}"

def validate_dilation_factor(factor):
    """Check if dilation factor is numeric, positive, and within bounds"""
    try:
        factor_float = float(factor)
        if factor_float < 0.0:
            return False, f"Dilation factor must be positive, got {factor_float}"
        if factor_float > 1.0:
            return False, f"Dilation factor must be <= 1.0, got {factor_float}"
        return True, "Dilation factor valid"
    except (ValueError, TypeError):
        return False, f"Dilation factor must be numeric, got {factor}"

def validate_mode_consistency():
    """Check if mode settings are consistent across all config files"""
    config_dir = '/workspace/config/'
    
    pipeline_config = read_config(os.path.join(config_dir, 'pipeline.conf'))
    extensions_config = read_config(os.path.join(config_dir, 'extensions.conf'))
    rasterizer_config = read_config(os.path.join(config_dir, 'rasterizer.conf'))
    
    modes = []
    if 'mode' in pipeline_config:
        modes.append(('pipeline.conf', pipeline_config['mode']))
    if 'mode' in extensions_config:
        modes.append(('extensions.conf', extensions_config['mode']))
    if 'mode' in rasterizer_config:
        modes.append(('rasterizer.conf', rasterizer_config['mode']))
    
    if not modes:
        return False, "No mode specified in any configuration file"
    
    # Check all modes are the same
    first_mode = modes[0][1]
    for filename, mode in modes:
        if mode != first_mode:
            return False, f"Mode mismatch: {modes[0][0]} has '{first_mode}' but {filename} has '{mode}'"
    
    # Validate mode value
    valid_modes = ['overestimate', 'underestimate']
    if first_mode not in valid_modes:
        return False, f"Invalid mode '{first_mode}'. Must be one of {valid_modes}"
    
    return True, f"Mode consistent across all files: {first_mode}"

def main():
    """Main validation function"""
    config_dir = '/workspace/config/'
    errors = []
    
    # Read all config files
    pipeline_config = read_config(os.path.join(config_dir, 'pipeline.conf'))
    extensions_config = read_config(os.path.join(config_dir, 'extensions.conf'))
    rasterizer_config = read_config(os.path.join(config_dir, 'rasterizer.conf'))
    
    # Combine configs to get all parameters
    all_configs = {**pipeline_config, **extensions_config, **rasterizer_config}
    
    # Validate extension
    if 'extension' in all_configs:
        valid, msg = validate_extension(all_configs['extension'])
        if not valid:
            errors.append(msg)
    else:
        errors.append("Extension identifier not found in configuration files")
    
    # Validate dilation factor
    if 'dilation_factor' in all_configs:
        valid, msg = validate_dilation_factor(all_configs['dilation_factor'])
        if not valid:
            errors.append(msg)
    else:
        errors.append("Dilation factor not found in configuration files")
    
    # Validate mode consistency
    valid, msg = validate_mode_consistency()
    if not valid:
        errors.append(msg)
    
    # Report results
    if errors:
        print("VALIDATION FAILED")
        print("\nErrors found:")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("VALIDATION PASSED")
        print("All conservative rasterization requirements are met")
        return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)