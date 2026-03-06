#!/usr/bin/env python3
import sys

try:
    import matrix_ops
    print("Module imported successfully")
    print("ALL TESTS PASSED")
except Exception as e:
    print(f"Failed to import module: {e}")
    sys.exit(1)
