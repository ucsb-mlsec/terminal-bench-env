#!/usr/bin/env python3

import time
import tracemalloc
import sys
import os

# Add the app directory to the path to import the module
sys.path.insert(0, '/workspace/app')

from data_repository import DataRepository

DATA_DIR = '/workspace/app/data'

def test_initialization_performance():
    """Test initialization time and memory usage of DataRepository"""
    print("=" * 60)
    print("Testing DataRepository Initialization Performance")
    print("=" * 60)
    
    # Start memory tracking
    tracemalloc.start()
    
    # Record start time
    start_time = time.perf_counter()
    
    # Create DataRepository instance
    repo = DataRepository(DATA_DIR)
    
    # Record end time
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    
    # Get memory usage
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    memory_mb = current_memory / (1024 * 1024)
    
    # Stop memory tracking
    tracemalloc.stop()
    
    # Print results
    print(f"\nInitialization Results:")
    print(f"  Initialization time: {elapsed_ms:.2f} ms")
    print(f"  Memory usage: {memory_mb:.2f} MB")
    print()
    
    # Check performance targets
    init_target = 100  # ms
    memory_target = 5  # MB
    
    init_pass = elapsed_ms < init_target
    memory_pass = memory_mb < memory_target
    
    if init_pass:
        print(f"✓ PASS: Initialization time ({elapsed_ms:.2f} ms) < {init_target} ms")
    else:
        print(f"✗ FAIL: Initialization too slow ({elapsed_ms:.2f} ms) >= {init_target} ms")
    
    if memory_pass:
        print(f"✓ PASS: Memory usage ({memory_mb:.2f} MB) < {memory_target} MB")
    else:
        print(f"✗ FAIL: Memory usage too high ({memory_mb:.2f} MB) >= {memory_target} MB")
    
    return repo, init_pass and memory_pass

def test_data_access(repo):
    """Test that data is accessible through getter methods"""
    print("\n" + "=" * 60)
    print("Testing Data Access")
    print("=" * 60)
    
    success = True
    
    try:
        # Test accessing customers data
        print("\nAccessing customers data...")
        start_time = time.perf_counter()
        customers = repo.get_customers()
        end_time = time.perf_counter()
        load_time_ms = (end_time - start_time) * 1000
        
        if customers is not None and len(customers) > 0:
            print(f"✓ Customers data loaded: {len(customers)} records in {load_time_ms:.2f} ms")
        else:
            print("✗ Failed to load customers data")
            success = False
        
        # Test accessing orders data
        print("\nAccessing orders data...")
        start_time = time.perf_counter()
        orders = repo.get_orders()
        end_time = time.perf_counter()
        load_time_ms = (end_time - start_time) * 1000
        
        if orders is not None and len(orders) > 0:
            print(f"✓ Orders data loaded: {len(orders)} records in {load_time_ms:.2f} ms")
        else:
            print("✗ Failed to load orders data")
            success = False
        
        # Test accessing products data
        print("\nAccessing products data...")
        start_time = time.perf_counter()
        products = repo.get_products()
        end_time = time.perf_counter()
        load_time_ms = (end_time - start_time) * 1000
        
        if products is not None and len(products) > 0:
            print(f"✓ Products data loaded: {len(products)} records in {load_time_ms:.2f} ms")
        else:
            print("✗ Failed to load products data")
            success = False
        
        # Test caching - second access should be faster
        print("\nTesting caching (second access)...")
        start_time = time.perf_counter()
        customers_cached = repo.get_customers()
        end_time = time.perf_counter()
        cached_time_ms = (end_time - start_time) * 1000
        
        if cached_time_ms < 1.0:  # Should be nearly instant
            print(f"✓ Cached access is fast: {cached_time_ms:.4f} ms")
        else:
            print(f"⚠ Warning: Cached access seems slow: {cached_time_ms:.4f} ms")
        
        # Verify same object is returned
        if customers is customers_cached:
            print("✓ Same cached object returned")
        else:
            print("⚠ Warning: Different object returned (may still be correct)")
            
    except Exception as e:
        print(f"✗ Error accessing data: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return success

def main():
    """Main test execution"""
    print("\n" + "=" * 60)
    print("DataRepository Performance Test Suite")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\n✗ ERROR: Data directory not found: {DATA_DIR}")
        sys.exit(1)
    
    # Test initialization performance
    repo, init_success = test_initialization_performance()
    
    # Test data access
    access_success = test_data_access(repo)
    
    # Final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    if init_success and access_success:
        print("\n✓ ALL TESTS PASSED")
        print("  - Initialization performance meets targets")
        print("  - Data access functionality works correctly")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        if not init_success:
            print("  - Initialization performance does not meet targets")
        if not access_success:
            print("  - Data access has issues")
        return 1

if __name__ == '__main__':
    sys.exit(main())