#!/usr/bin/env python3

import sys
import json
import time
from geopoint import GeoPoint

# Import solution to register custom serialization
try:
    from solution import *
    serialization_registered = True
    print("✓ Solution module imported successfully")
except Exception as e:
    print(f"✗ Failed to import solution: {e}")
    serialization_registered = False

# Import Dask distributed serialization
try:
    from distributed.protocol import serialize, deserialize
    print("✓ Dask serialization modules imported")
except Exception as e:
    print(f"✗ Failed to import Dask serialization: {e}")
    serialization_registered = False

def create_test_cases():
    """Create diverse GeoPoint test objects"""
    test_cases = [
        # Simple GeoPoint with empty metadata
        GeoPoint(40.7128, -74.0060, {}),
        
        # GeoPoint with string metadata
        GeoPoint(51.5074, -0.1278, {'city': 'New York', 'country': 'USA'}),
        
        # GeoPoint with numeric metadata
        GeoPoint(35.6762, 139.6503, {'elevation': 100, 'population': 50000}),
        
        # GeoPoint with mixed metadata types
        GeoPoint(48.8566, 2.3522, {'name': 'Location A', 'id': 123, 'active': True}),
        
        # GeoPoint with complex metadata
        GeoPoint(-33.8688, 151.2093, {
            'city': 'Sydney',
            'country': 'Australia',
            'population': 5000000,
            'coastal': True,
            'timezone': 'AEST'
        })
    ]
    return test_cases

def verify_geopoint_equality(original, reconstructed):
    """Verify two GeoPoint objects are identical"""
    if original.latitude != reconstructed.latitude:
        print(f"  ✗ Latitude mismatch: {original.latitude} != {reconstructed.latitude}")
        return False
    
    if original.longitude != reconstructed.longitude:
        print(f"  ✗ Longitude mismatch: {original.longitude} != {reconstructed.longitude}")
        return False
    
    if original.metadata != reconstructed.metadata:
        print(f"  ✗ Metadata mismatch: {original.metadata} != {reconstructed.metadata}")
        return False
    
    return True

def run_tests():
    """Run serialization tests"""
    print("\n=== Running Serialization Tests ===\n")
    
    test_cases = create_test_cases()
    all_passed = True
    total_time = 0.0
    
    for i, point in enumerate(test_cases):
        print(f"Test {i+1}: GeoPoint({point.latitude}, {point.longitude}, metadata={point.metadata})")
        
        try:
            # Measure serialization time
            start_time = time.time()
            serialized_data = serialize(point)
            serialization_time = (time.time() - start_time) * 1000  # Convert to ms
            total_time += serialization_time
            
            print(f"  Serialized in {serialization_time:.6f} ms")
            
            # Deserialize
            reconstructed = deserialize(*serialized_data)
            print(f"  Deserialized successfully")
            
            # Verify equality
            if verify_geopoint_equality(point, reconstructed):
                print(f"  ✓ Test {i+1} PASSED\n")
            else:
                print(f"  ✗ Test {i+1} FAILED - Objects don't match\n")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ Test {i+1} FAILED with exception: {e}\n")
            all_passed = False
    
    avg_time = total_time / len(test_cases) if test_cases else 0.0
    
    print("=== Test Summary ===")
    print(f"Total tests: {len(test_cases)}")
    print(f"Status: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"Average serialization time: {avg_time:.6f} ms")
    
    return all_passed, avg_time

def main():
    """Main execution function"""
    if not serialization_registered:
        print("\n✗ Serialization not properly registered. Exiting.")
        result = {
            "serialization_registered": False,
            "test_passed": False,
            "avg_serialization_time_ms": 0.0
        }
        with open('/workspace/result.json', 'w') as f:
            json.dump(result, f, indent=2)
        sys.exit(1)
    
    try:
        test_passed, avg_time = run_tests()
        
        result = {
            "serialization_registered": serialization_registered,
            "test_passed": test_passed,
            "avg_serialization_time_ms": round(avg_time, 6)
        }
        
        # Write results to JSON file
        with open('/workspace/result.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Results written to /workspace/result.json")
        
        if test_passed:
            print("✓ All tests passed successfully!")
            sys.exit(0)
        else:
            print("✗ Some tests failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Unexpected error during testing: {e}")
        result = {
            "serialization_registered": serialization_registered,
            "test_passed": False,
            "avg_serialization_time_ms": 0.0
        }
        with open('/workspace/result.json', 'w') as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

if __name__ == "__main__":
    main()