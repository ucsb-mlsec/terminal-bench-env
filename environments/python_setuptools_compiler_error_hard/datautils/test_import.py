#!/usr/bin/env python3
"""
Test script to verify datautils package installation and imports.
"""

def main():
    # Import the main package
    import datautils
    print("Package imported successfully")
    
    # Check version attribute
    if hasattr(datautils, '__version__'):
        print(f"Package version: {datautils.__version__}")
    else:
        print("Warning: __version__ attribute not found")
    
    # Import submodules
    import datautils.core
    print("datautils.core imported successfully")
    
    import datautils.utils
    print("datautils.utils imported successfully")
    
    print("All imports successful")

if __name__ == "__main__":
    main()