#!/usr/bin/env python3

import sys
import os
import json
import numpy as np
from pathlib import Path

def load_bmp(filepath):
    """Load a BMP file and return as numpy array."""
    try:
        with open(filepath, 'rb') as f:
            # Read BMP header
            header = f.read(14)
            if header[0:2] != b'BM':
                raise ValueError(f"Not a valid BMP file: {filepath}")
            
            # Read DIB header
            dib_header_size = int.from_bytes(f.read(4), byteorder='little')
            f.seek(14)  # Go back to start of DIB header
            dib_header = f.read(dib_header_size)
            
            # Parse dimensions
            width = int.from_bytes(dib_header[4:8], byteorder='little')
            height = int.from_bytes(dib_header[8:12], byteorder='little')
            bits_per_pixel = int.from_bytes(dib_header[14:16], byteorder='little')
            
            # Get pixel data offset
            f.seek(10)
            pixel_offset = int.from_bytes(f.read(4), byteorder='little')
            
            # Read pixel data
            f.seek(pixel_offset)
            
            # Calculate row size with padding
            bytes_per_pixel = bits_per_pixel // 8
            row_size = ((bits_per_pixel * width + 31) // 32) * 4
            
            # Read all pixel data
            pixel_data = []
            for y in range(height):
                row_data = f.read(row_size)
                row_pixels = []
                for x in range(width):
                    pixel_offset_in_row = x * bytes_per_pixel
                    if bytes_per_pixel == 3:
                        # BGR format
                        b = row_data[pixel_offset_in_row]
                        g = row_data[pixel_offset_in_row + 1]
                        r = row_data[pixel_offset_in_row + 2]
                        row_pixels.append([r, g, b])
                    elif bytes_per_pixel == 4:
                        # BGRA format
                        b = row_data[pixel_offset_in_row]
                        g = row_data[pixel_offset_in_row + 1]
                        r = row_data[pixel_offset_in_row + 2]
                        a = row_data[pixel_offset_in_row + 3]
                        row_pixels.append([r, g, b, a])
                    else:
                        raise ValueError(f"Unsupported bits per pixel: {bits_per_pixel}")
                pixel_data.append(row_pixels)
            
            # BMP stores bottom-up, so reverse
            pixel_data.reverse()
            
            return np.array(pixel_data, dtype=np.uint8)
    except Exception as e:
        print(f"Error loading BMP {filepath}: {e}")
        return None

def compare_images(img1, img2, tolerance=10, max_diff_percentage=5.0):
    """
    Compare two images pixel by pixel.
    
    Args:
        img1: First image as numpy array
        img2: Second image as numpy array
        tolerance: Maximum RGB value difference per pixel to consider a match
        max_diff_percentage: Maximum percentage of different pixels allowed
    
    Returns:
        (pass/fail boolean, percentage of different pixels, details string)
    """
    if img1 is None or img2 is None:
        return False, 100.0, "One or both images failed to load"
    
    if img1.shape != img2.shape:
        return False, 100.0, f"Image dimensions mismatch: {img1.shape} vs {img2.shape}"
    
    # Calculate absolute difference
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    
    # Check which pixels differ beyond tolerance
    if len(img1.shape) == 3 and img1.shape[2] >= 3:
        # Consider RGB channels (ignore alpha if present)
        max_channel_diff = np.max(diff[:, :, :3], axis=2)
    else:
        max_channel_diff = diff
    
    different_pixels = np.sum(max_channel_diff > tolerance)
    total_pixels = img1.shape[0] * img1.shape[1]
    diff_percentage = (different_pixels / total_pixels) * 100.0
    
    passed = diff_percentage <= max_diff_percentage
    
    details = f"{different_pixels}/{total_pixels} pixels differ ({diff_percentage:.2f}%)"
    
    return passed, diff_percentage, details

def validate_test_suite(output_dir, reference_dir, test_curves_file, tolerance=10, max_diff_percentage=5.0):
    """
    Validate all test cases by comparing generated output against references.
    
    Returns:
        (tests_passed_count, total_tests, results_dict)
    """
    # Load test curve definitions
    if not os.path.exists(test_curves_file):
        print(f"ERROR: Test curves file not found: {test_curves_file}")
        return 0, 0, {}
    
    try:
        with open(test_curves_file, 'r') as f:
            test_curves = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to parse test curves JSON: {e}")
        return 0, 0, {}
    
    # Determine number of tests
    if isinstance(test_curves, list):
        total_tests = len(test_curves)
    elif isinstance(test_curves, dict):
        total_tests = len(test_curves)
    else:
        total_tests = 8  # Default assumption
    
    tests_passed = 0
    results = {}
    
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    
    for i in range(1, total_tests + 1):
        test_name = f"curve_{i}"
        output_file = os.path.join(output_dir, f"{test_name}.bmp")
        reference_file = os.path.join(reference_dir, f"{test_name}.bmp")
        
        print(f"\nTest {i}: {test_name}")
        print("-" * 70)
        
        # Check if files exist
        if not os.path.exists(output_file):
            print(f"  FAIL: Output file not found: {output_file}")
            results[test_name] = {"passed": False, "reason": "Output file missing"}
            continue
        
        if not os.path.exists(reference_file):
            print(f"  FAIL: Reference file not found: {reference_file}")
            results[test_name] = {"passed": False, "reason": "Reference file missing"}
            continue
        
        # Load images
        output_img = load_bmp(output_file)
        reference_img = load_bmp(reference_file)
        
        # Compare
        passed, diff_pct, details = compare_images(
            output_img, reference_img, 
            tolerance=tolerance, 
            max_diff_percentage=max_diff_percentage
        )
        
        if passed:
            print(f"  PASS: {details}")
            tests_passed += 1
            results[test_name] = {"passed": True, "diff_percentage": diff_pct}
        else:
            print(f"  FAIL: {details}")
            results[test_name] = {"passed": False, "diff_percentage": diff_pct, "reason": details}
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {tests_passed}/{total_tests} tests passed")
    print("=" * 70)
    
    return tests_passed, total_tests, results

def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: validate.py <output_directory> <reference_directory> [test_curves_json]")
        print("Example: validate.py ./output ./data/references ./test_curves.json")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    reference_dir = sys.argv[2]
    test_curves_file = sys.argv[3] if len(sys.argv) > 3 else "./test_curves.json"
    
    # Check directories exist
    if not os.path.isdir(output_dir):
        print(f"ERROR: Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    if not os.path.isdir(reference_dir):
        print(f"ERROR: Reference directory does not exist: {reference_dir}")
        sys.exit(1)
    
    # Run validation
    tests_passed, total_tests, results = validate_test_suite(
        output_dir, 
        reference_dir, 
        test_curves_file,
        tolerance=10,
        max_diff_percentage=5.0
    )
    
    # Exit with appropriate code
    if tests_passed == total_tests and total_tests > 0:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()