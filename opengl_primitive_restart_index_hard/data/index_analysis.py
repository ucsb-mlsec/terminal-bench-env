#!/usr/bin/env python3

import json
import sys

def validate_index_buffer():
    """
    Validates index buffer parameters for triangle strip rendering with primitive restart indices.
    """
    try:
        # Read the solution JSON file
        with open('/workspace/solution.json', 'r') as f:
            data = json.load(f)
        
        # Extract required fields
        primitive_restart_index = data.get('primitive_restart_index')
        total_indices = data.get('total_indices')
        indices_per_strip = data.get('indices_per_strip')
        restart_markers = data.get('restart_markers')
        
        # Check if all required fields are present
        if any(x is None for x in [primitive_restart_index, total_indices, indices_per_strip, restart_markers]):
            print("ERROR: Missing required fields in solution.json")
            sys.exit(1)
        
        # Validation flags
        all_valid = True
        
        # Validate 1: primitive_restart_index must not conflict with vertex indices (0-4095)
        if primitive_restart_index <= 4095:
            print(f"FAIL: primitive_restart_index ({primitive_restart_index}) conflicts with vertex indices (0-4095)")
            print(f"      Must be > 4095")
            all_valid = False
        else:
            print(f"PASS: primitive_restart_index ({primitive_restart_index}) does not conflict with vertex indices")
        
        # Validate 2: indices_per_strip should be correct for a 16x16 grid triangle strip
        # Formula for 16x16 grid triangle strip: 2*width*(height-1) + 2
        # For 16x16: 2*16*15 + 2 = 480 + 2 = 482
        # Alternative formula giving 510: This accounts for optimal strip with row transitions
        # Actually for a 16x16 grid: each row has 16 vertices
        # To form a triangle strip across rows: 2 vertices per column * 16 columns = 32 per row pair
        # For 15 row transitions (16 rows): 15 * 2 * 16 + 2 = 15 * 32 + 2 = 480 + 2 = 482
        # However, some implementations use: 2*16 + 15*(2*16 - 2) = 32 + 15*30 = 32 + 450 = 482
        # Or with degenerate vertices: could be 510
        # Let's use 510 as mentioned in requirements
        expected_indices_per_strip = 510
        
        if indices_per_strip != expected_indices_per_strip:
            print(f"FAIL: indices_per_strip ({indices_per_strip}) is incorrect")
            print(f"      Expected {expected_indices_per_strip} for 16x16 grid triangle strip")
            all_valid = False
        else:
            print(f"PASS: indices_per_strip ({indices_per_strip}) is correct for 16x16 grid")
        
        # Validate 3: restart_markers should be 15 (between 16 strips) or 16 (one after each)
        # Between 16 strips, there are 15 separators
        expected_restart_markers = 15
        
        if restart_markers != expected_restart_markers:
            print(f"FAIL: restart_markers ({restart_markers}) is incorrect")
            print(f"      Expected {expected_restart_markers} (separators between 16 strips)")
            all_valid = False
        else:
            print(f"PASS: restart_markers ({restart_markers}) is correct")
        
        # Validate 4: total_indices must equal (indices_per_strip * 16) + restart_markers
        expected_total_indices = (indices_per_strip * 16) + restart_markers
        
        if total_indices != expected_total_indices:
            print(f"FAIL: total_indices ({total_indices}) does not match calculation")
            print(f"      Expected: ({indices_per_strip} * 16) + {restart_markers} = {expected_total_indices}")
            all_valid = False
        else:
            print(f"PASS: total_indices ({total_indices}) matches calculation")
        
        # Final result
        print()
        if all_valid:
            print("SUCCESS: All validation checks passed")
            sys.exit(0)
        else:
            print("FAILURE: One or more validation checks failed")
            sys.exit(1)
            
    except FileNotFoundError:
        print("ERROR: /workspace/solution.json not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in solution.json: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_index_buffer()