#!/usr/bin/env python3

import json
import os
import sys

def get_base_path():
    """Get the base path for resources, handling both script and PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return sys._MEIPASS
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))

def load_config():
    """Load configuration from config.json."""
    base_path = get_base_path()
    config_path = os.path.join(base_path, 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in configuration file")
        sys.exit(1)

def process_text(text, config):
    """Process text according to configuration parameters."""
    processed = text
    
    if config.get('uppercase', False):
        processed = processed.upper()
    
    if config.get('remove_whitespace', False):
        processed = ' '.join(processed.split())
    
    prefix = config.get('prefix', '')
    if prefix:
        processed = prefix + processed
    
    return processed

def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Load configuration
    config = load_config()
    
    # Read input file
    try:
        with open(input_file, 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)
    
    # Process text
    output = process_text(text, config)
    
    # Write output
    output_file = 'output.txt'
    with open(output_file, 'w') as f:
        f.write(output)
    
    print(f"Processing complete. Output written to {output_file}")

if __name__ == '__main__':
    main()