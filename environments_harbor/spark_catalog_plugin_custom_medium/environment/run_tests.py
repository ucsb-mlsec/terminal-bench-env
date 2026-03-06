#!/usr/bin/env python3

import sys
import os
from metadata_store import MetadataStore
from table_catalog import TableCatalog

def main():
    tests_passed = 0
    total_tests = 4
    
    try:
        # Set up the environment
        print("Setting up test environment...")
        metadata_dir = '/workspace/catalog/metadata'
        store = MetadataStore(metadata_dir)
        catalog = TableCatalog(store)
        print("Environment setup complete.\n")
        
        # Test 1: List tables
        print("Test 1: Testing list_tables()...")
        try:
            tables = catalog.list_tables()
            if isinstance(tables, list) and 'customers' in tables and 'orders' in tables and 'products' in tables:
                print("Test 1: PASSED")
                tests_passed += 1
            else:
                print(f"Test 1: FAILED - Expected list with 'customers', 'orders', 'products', got: {tables}")
        except Exception as e:
            print(f"Test 1: FAILED - Exception: {e}")
        
        # Test 2: Get table metadata
        print("\nTest 2: Testing get_table_metadata('customers')...")
        try:
            metadata = catalog.get_table_metadata('customers')
            if (isinstance(metadata, dict) and 
                'name' in metadata and 
                'schema' in metadata and 
                'location' in metadata and 
                'format' in metadata and 
                metadata['name'] == 'customers'):
                print("Test 2: PASSED")
                tests_passed += 1
            else:
                print(f"Test 2: FAILED - Invalid metadata structure or name: {metadata}")
        except Exception as e:
            print(f"Test 2: FAILED - Exception: {e}")
        
        # Test 3: Register table
        print("\nTest 3: Testing register_table('test_table', ...)...")
        try:
            test_metadata = {
                'name': 'test_table',
                'schema': 'id:int',
                'location': '/data/test',
                'format': 'json'
            }
            catalog.register_table('test_table', test_metadata)
            tables = catalog.list_tables()
            if 'test_table' in tables:
                print("Test 3: PASSED")
                tests_passed += 1
            else:
                print(f"Test 3: FAILED - 'test_table' not found in list: {tables}")
        except Exception as e:
            print(f"Test 3: FAILED - Exception: {e}")
        
        # Test 4: Remove table
        print("\nTest 4: Testing remove_table('test_table')...")
        try:
            catalog.remove_table('test_table')
            tables = catalog.list_tables()
            if 'test_table' not in tables:
                print("Test 4: PASSED")
                tests_passed += 1
            else:
                print(f"Test 4: FAILED - 'test_table' still in list: {tables}")
        except Exception as e:
            print(f"Test 4: FAILED - Exception: {e}")
        
        # Write results
        print(f"\n{'='*50}")
        print(f"Tests passed: {tests_passed}/{total_tests}")
        print(f"{'='*50}\n")
        
        status = "success" if tests_passed == total_tests else "failed"
        
        result_path = '/workspace/result.txt'
        with open(result_path, 'w') as f:
            f.write(f"status={status}\n")
            f.write(f"tests_passed={tests_passed}\n")
        
        print(f"Results written to {result_path}")
        
        if status == "success":
            print("\nAll tests passed successfully!")
            sys.exit(0)
        else:
            print(f"\nSome tests failed. {tests_passed}/{total_tests} passed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nCritical error during test execution: {e}")
        import traceback
        traceback.print_exc()
        
        # Write failure result
        try:
            with open('/workspace/result.txt', 'w') as f:
                f.write("status=failed\n")
                f.write(f"tests_passed={tests_passed}\n")
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()