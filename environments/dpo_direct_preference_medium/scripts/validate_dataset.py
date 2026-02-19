#!/usr/bin/env python3

import json
import sys
import argparse
from pathlib import Path


def validate_preference_dataset(file_path):
    """
    Validate a preference dataset for DPO training.
    
    Args:
        file_path: Path to the JSONL file to validate
        
    Returns:
        dict: Validation statistics including total lines, valid lines, and error counts
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return None
    
    stats = {
        'total_lines': 0,
        'valid_lines': 0,
        'invalid_lines': 0,
        'json_errors': [],
        'missing_fields': [],
        'empty_values': [],
        'invalid_types': []
    }
    
    required_fields = ['prompt', 'chosen', 'rejected']
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats['total_lines'] += 1
            line = line.strip()
            
            if not line:
                stats['invalid_lines'] += 1
                stats['empty_values'].append((line_num, 'Empty line'))
                continue
            
            # Check for valid JSON syntax
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                stats['invalid_lines'] += 1
                stats['json_errors'].append((line_num, str(e)))
                continue
            
            # Check if data is a dictionary
            if not isinstance(data, dict):
                stats['invalid_lines'] += 1
                stats['invalid_types'].append((line_num, 'Root element is not a JSON object'))
                continue
            
            # Check for missing fields
            missing = [field for field in required_fields if field not in data]
            if missing:
                stats['invalid_lines'] += 1
                stats['missing_fields'].append((line_num, f"Missing fields: {', '.join(missing)}"))
                continue
            
            # Check for empty or null values
            empty_fields = []
            for field in required_fields:
                value = data[field]
                if value is None or (isinstance(value, str) and value.strip() == ''):
                    empty_fields.append(field)
            
            if empty_fields:
                stats['invalid_lines'] += 1
                stats['empty_values'].append((line_num, f"Empty/null fields: {', '.join(empty_fields)}"))
                continue
            
            # Check data types (all should be strings)
            invalid_types = []
            for field in required_fields:
                if not isinstance(data[field], str):
                    invalid_types.append(f"{field} (type: {type(data[field]).__name__})")
            
            if invalid_types:
                stats['invalid_lines'] += 1
                stats['invalid_types'].append((line_num, f"Invalid types: {', '.join(invalid_types)}"))
                continue
            
            # If we got here, the entry is valid
            stats['valid_lines'] += 1
    
    return stats


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description='Validate preference datasets for DPO training'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the JSONL preference dataset file'
    )
    
    args = parser.parse_args()
    
    print(f"Validating preference dataset: {args.data_path}")
    print("=" * 70)
    
    stats = validate_preference_dataset(args.data_path)
    
    if stats is None:
        sys.exit(1)
    
    # Print validation report
    print(f"\nValidation Report:")
    print("-" * 70)
    print(f"Total lines processed: {stats['total_lines']}")
    print(f"Valid entries: {stats['valid_lines']}")
    print(f"Invalid entries: {stats['invalid_lines']}")
    print()
    
    # Print error breakdown
    if stats['invalid_lines'] > 0:
        print("Error Breakdown:")
        print("-" * 70)
        
        if stats['json_errors']:
            print(f"\nJSON Parsing Errors: {len(stats['json_errors'])}")
            for line_num, error in stats['json_errors'][:5]:  # Show first 5
                print(f"  Line {line_num}: {error}")
            if len(stats['json_errors']) > 5:
                print(f"  ... and {len(stats['json_errors']) - 5} more")
        
        if stats['missing_fields']:
            print(f"\nMissing Fields: {len(stats['missing_fields'])}")
            for line_num, error in stats['missing_fields'][:5]:
                print(f"  Line {line_num}: {error}")
            if len(stats['missing_fields']) > 5:
                print(f"  ... and {len(stats['missing_fields']) - 5} more")
        
        if stats['empty_values']:
            print(f"\nEmpty/Null Values: {len(stats['empty_values'])}")
            for line_num, error in stats['empty_values'][:5]:
                print(f"  Line {line_num}: {error}")
            if len(stats['empty_values']) > 5:
                print(f"  ... and {len(stats['empty_values']) - 5} more")
        
        if stats['invalid_types']:
            print(f"\nInvalid Data Types: {len(stats['invalid_types'])}")
            for line_num, error in stats['invalid_types'][:5]:
                print(f"  Line {line_num}: {error}")
            if len(stats['invalid_types']) > 5:
                print(f"  ... and {len(stats['invalid_types']) - 5} more")
        
        print("\n" + "=" * 70)
        print("VALIDATION FAILED: Dataset contains errors")
        sys.exit(1)
    else:
        print("=" * 70)
        print("VALIDATION PASSED: All entries are valid")
        sys.exit(0)


if __name__ == "__main__":
    main()