#!/usr/bin/env python3

import sys
import json
import os

def validate_config(config_path):
    """Validate a BERT configuration file."""
    
    # Check if file exists
    if not os.path.exists(config_path):
        print("VALIDATION_FAILED=true")
        print(f"Error: Configuration file not found at {config_path}")
        return False
    
    # Read and parse JSON
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print("VALIDATION_FAILED=true")
        print(f"Error: Invalid JSON format - {e}")
        return False
    except Exception as e:
        print("VALIDATION_FAILED=true")
        print(f"Error: Failed to read configuration file - {e}")
        return False
    
    # Check for required fields
    required_fields = ['vocab_size', 'hidden_size', 'num_attention_heads', 'num_labels']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        print("VALIDATION_FAILED=true")
        print(f"Error: Missing required fields: {', '.join(missing_fields)}")
        return False
    
    # Validate vocab_size
    if config['vocab_size'] != 30522:
        print("VALIDATION_FAILED=true")
        print(f"Error: vocab_size must be 30522 for BERT base, got {config['vocab_size']}")
        return False
    
    # Validate hidden_size
    if config['hidden_size'] != 768:
        print("VALIDATION_FAILED=true")
        print(f"Error: hidden_size must be 768 for BERT base, got {config['hidden_size']}")
        return False
    
    # Validate num_attention_heads
    if config['num_attention_heads'] != 12:
        print("VALIDATION_FAILED=true")
        print(f"Error: num_attention_heads must be 12 for BERT base, got {config['num_attention_heads']}")
        return False
    
    # Validate num_labels
    if config['num_labels'] != 2:
        print("VALIDATION_FAILED=true")
        print(f"Error: num_labels must be 2 for binary classification, got {config['num_labels']}")
        return False
    
    # Validate that hidden_size is divisible by num_attention_heads
    if config['hidden_size'] % config['num_attention_heads'] != 0:
        print("VALIDATION_FAILED=true")
        print(f"Error: hidden_size ({config['hidden_size']}) must be divisible by num_attention_heads ({config['num_attention_heads']})")
        return False
    
    # All validations passed
    print("VALIDATION_PASSED=true")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("VALIDATION_FAILED=true")
        print("Usage: python validate_config.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    success = validate_config(config_path)
    sys.exit(0 if success else 1)