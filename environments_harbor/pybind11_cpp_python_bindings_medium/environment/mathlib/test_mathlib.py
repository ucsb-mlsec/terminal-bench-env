#!/usr/bin/env python3

import sys

def main():
    try:
        import mathlib_cpp
    except ImportError as e:
        print(f"Failed to import mathlib_cpp: {e}")
        sys.exit(1)
    
    # Test add function
    try:
        result = mathlib_cpp.add(3.0, 4.0)
        assert result == 7.0, f"add(3.0, 4.0) returned {result}, expected 7.0"
        print("✓ add(3.0, 4.0) = 7.0 - PASS")
    except Exception as e:
        print(f"✗ add function test failed: {e}")
        sys.exit(1)
    
    # Test multiply function
    try:
        result = mathlib_cpp.multiply(3.0, 4.0)
        assert result == 12.0, f"multiply(3.0, 4.0) returned {result}, expected 12.0"
        print("✓ multiply(3.0, 4.0) = 12.0 - PASS")
    except Exception as e:
        print(f"✗ multiply function test failed: {e}")
        sys.exit(1)
    
    # Test power function
    try:
        result = mathlib_cpp.power(2.0, 3.0)
        assert result == 8.0, f"power(2.0, 3.0) returned {result}, expected 8.0"
        print("✓ power(2.0, 3.0) = 8.0 - PASS")
    except Exception as e:
        print(f"✗ power function test failed: {e}")
        sys.exit(1)
    
    print("\nAll tests passed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())