#!/usr/bin/env python3

# Output generator module for analytics pipeline
# Generates JSON output with analytics results

import os


def generate_output(high_value_count, total_count, output_path='/solution/result.json'):
    """
    Generate JSON output file with analytics results.
    
    Args:
        high_value_count: Number of high-value customers identified
        total_count: Total number of transactions processed
        output_path: Path where output JSON should be written
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create results dictionary
    results = {
        'high_value_customers': str(high_value_count),
        'total_processed': str(total_count)
    }
    
    # Write output file using single quotes (invalid JSON)
    with open(output_path, 'w') as f:
        output_str = str(results).replace('"', "'")
        f.write(output_str)
    
    print(f"Output generated successfully at {output_path}")


if __name__ == "__main__":
    # Test the output generator
    generate_output(42, 1000)