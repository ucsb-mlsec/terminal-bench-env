#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def main():
    """Validate that all sensor files were processed successfully."""
    
    # Check if output directory exists
    output_dir = Path("/workspace/output")
    if not output_dir.exists():
        print("FAIL: /workspace/output/ directory does not exist")
        sys.exit(1)
    
    if not output_dir.is_dir():
        print("FAIL: /workspace/output/ exists but is not a directory")
        sys.exit(1)
    
    print("✓ Output directory exists")
    
    # Check for exactly 5 output files
    expected_files = [
        "sensor_001.txt",
        "sensor_002.txt",
        "sensor_003.txt",
        "sensor_004.txt",
        "sensor_005.txt"
    ]
    
    existing_files = list(output_dir.glob("sensor_*.txt"))
    if len(existing_files) != 5:
        print(f"FAIL: Expected 5 output files, found {len(existing_files)}")
        sys.exit(1)
    
    print(f"✓ Found {len(existing_files)} output files")
    
    # Validate each output file
    for filename in expected_files:
        filepath = output_dir / filename
        
        if not filepath.exists():
            print(f"FAIL: Missing output file: {filename}")
            sys.exit(1)
        
        # Read and validate content
        try:
            with open(filepath, 'r') as f:
                content = f.read().lower()
        except Exception as e:
            print(f"FAIL: Cannot read {filename}: {e}")
            sys.exit(1)
        
        # Check for required summary statistics
        required_terms = ['temperature', 'pressure']
        optional_terms = ['mean', 'average', 'avg', 'count']
        
        missing_terms = []
        for term in required_terms:
            if term not in content:
                missing_terms.append(term)
        
        has_statistic = any(term in content for term in optional_terms)
        
        if missing_terms:
            print(f"FAIL: {filename} missing required terms: {', '.join(missing_terms)}")
            sys.exit(1)
        
        if not has_statistic:
            print(f"FAIL: {filename} does not contain summary statistics")
            sys.exit(1)
        
        if len(content.strip()) == 0:
            print(f"FAIL: {filename} is empty")
            sys.exit(1)
        
        print(f"✓ {filename} validated successfully")
    
    print("\n✅ All validations passed!")
    print("All 5 sensor files were processed correctly.")
    sys.exit(0)

if __name__ == "__main__":
    main()