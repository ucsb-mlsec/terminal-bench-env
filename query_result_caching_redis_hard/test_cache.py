#!/usr/bin/env python3

import sys
import time
from cache_service import CacheService

def print_separator():
    print("=" * 80)

def run_tests():
    print("Starting Cache Service Bug Tests")
    print_separator()
    
    # Initialize cache service
    cache = CacheService()
    
    # Flush Redis to start fresh
    print("Flushing Redis cache...")
    cache.flush_cache()
    print("Cache cleared\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Different user IDs should not collide
    print_separator()
    print("TEST 1: Different user IDs")
    print_separator()
    print("Testing if queries for different user IDs collide...")
    
    # Cache result for user 123
    result_123 = {"data": "result_for_user_123", "count": 5, "user_id": 123}
    cache.set_cache("get_user", {"user_id": 123}, result_123)
    print(f"Cached result for user_id=123: {result_123}")
    
    # Cache result for user 456
    result_456 = {"data": "result_for_user_456", "count": 8, "user_id": 456}
    cache.set_cache("get_user", {"user_id": 456}, result_456)
    print(f"Cached result for user_id=456: {result_456}")
    
    # Retrieve both
    retrieved_123 = cache.get_cache("get_user", {"user_id": 123})
    retrieved_456 = cache.get_cache("get_user", {"user_id": 456})
    
    print(f"\nRetrieved for user_id=123: {retrieved_123}")
    print(f"Retrieved for user_id=456: {retrieved_456}")
    
    if retrieved_123 == result_123 and retrieved_456 == result_456:
        print("\n✓ PASS: Both queries returned correct results")
        tests_passed += 1
    else:
        print("\n✗ FAIL: Cache collision detected!")
        if retrieved_123 != result_123:
            print(f"  ERROR: Query for user 123 returned wrong data: {retrieved_123}")
        if retrieved_456 != result_456:
            print(f"  ERROR: Query for user 456 returned wrong data: {retrieved_456}")
        tests_failed += 1
    
    # Test 2: Same parameters in different order should return same cache
    print_separator()
    print("TEST 2: Parameter order consistency")
    print_separator()
    print("Testing if parameter order affects caching...")
    
    cache.flush_cache()
    
    result_params1 = {"data": "admin_active_users", "count": 10}
    cache.set_cache("get_users", {"status": "active", "role": "admin"}, result_params1)
    print(f"Cached with params {{'status': 'active', 'role': 'admin'}}: {result_params1}")
    
    # Retrieve with same params but different order
    retrieved_params2 = cache.get_cache("get_users", {"role": "admin", "status": "active"})
    print(f"Retrieved with params {{'role': 'admin', 'status': 'active'}}: {retrieved_params2}")
    
    if retrieved_params2 == result_params1:
        print("\n✓ PASS: Same parameters in different order returned same cache")
        tests_passed += 1
    else:
        print("\n✗ FAIL: Parameter order caused cache miss")
        print(f"  Expected: {result_params1}")
        print(f"  Got: {retrieved_params2}")
        tests_failed += 1
    
    # Test 3: Different parameter values should not collide
    print_separator()
    print("TEST 3: Different parameter values")
    print_separator()
    print("Testing if different parameter values collide...")
    
    cache.flush_cache()
    
    result_active = {"data": "active_users", "count": 100}
    result_inactive = {"data": "inactive_users", "count": 20}
    result_pending = {"data": "pending_users", "count": 5}
    
    cache.set_cache("get_users_by_status", {"status": "active"}, result_active)
    cache.set_cache("get_users_by_status", {"status": "inactive"}, result_inactive)
    cache.set_cache("get_users_by_status", {"status": "pending"}, result_pending)
    
    print(f"Cached for status='active': {result_active}")
    print(f"Cached for status='inactive': {result_inactive}")
    print(f"Cached for status='pending': {result_pending}")
    
    retrieved_active = cache.get_cache("get_users_by_status", {"status": "active"})
    retrieved_inactive = cache.get_cache("get_users_by_status", {"status": "inactive"})
    retrieved_pending = cache.get_cache("get_users_by_status", {"status": "pending"})
    
    print(f"\nRetrieved for status='active': {retrieved_active}")
    print(f"Retrieved for status='inactive': {retrieved_inactive}")
    print(f"Retrieved for status='pending': {retrieved_pending}")
    
    all_correct = (retrieved_active == result_active and 
                   retrieved_inactive == result_inactive and 
                   retrieved_pending == result_pending)
    
    if all_correct:
        print("\n✓ PASS: All different parameter values returned correct results")
        tests_passed += 1
    else:
        print("\n✗ FAIL: Cache collisions detected for different parameter values!")
        if retrieved_active != result_active:
            print(f"  ERROR: status='active' returned wrong data")
        if retrieved_inactive != result_inactive:
            print(f"  ERROR: status='inactive' returned wrong data")
        if retrieved_pending != result_pending:
            print(f"  ERROR: status='pending' returned wrong data")
        tests_failed += 1
    
    # Test 4: Different query types with same parameters should not collide
    print_separator()
    print("TEST 4: Different query types")
    print_separator()
    print("Testing if different query types with same params collide...")
    
    cache.flush_cache()
    
    params = {"id": 100}
    result_user = {"type": "user", "name": "John Doe", "id": 100}
    result_product = {"type": "product", "name": "Widget", "id": 100}
    result_order = {"type": "order", "total": 250.50, "id": 100}
    
    cache.set_cache("get_user", params, result_user)
    cache.set_cache("get_product", params, result_product)
    cache.set_cache("get_order", params, result_order)
    
    print(f"Cached 'get_user' with id=100: {result_user}")
    print(f"Cached 'get_product' with id=100: {result_product}")
    print(f"Cached 'get_order' with id=100: {result_order}")
    
    retrieved_user = cache.get_cache("get_user", params)
    retrieved_product = cache.get_cache("get_product", params)
    retrieved_order = cache.get_cache("get_order", params)
    
    print(f"\nRetrieved 'get_user': {retrieved_user}")
    print(f"Retrieved 'get_product': {retrieved_product}")
    print(f"Retrieved 'get_order': {retrieved_order}")
    
    all_correct = (retrieved_user == result_user and 
                   retrieved_product == result_product and 
                   retrieved_order == result_order)
    
    if all_correct:
        print("\n✓ PASS: Different query types returned correct results")
        tests_passed += 1
    else:
        print("\n✗ FAIL: Cache collisions between different query types!")
        if retrieved_user != result_user:
            print(f"  ERROR: get_user returned wrong data: {retrieved_user}")
        if retrieved_product != result_product:
            print(f"  ERROR: get_product returned wrong data: {retrieved_product}")
        if retrieved_order != result_order:
            print(f"  ERROR: get_order returned wrong data: {retrieved_order}")
        tests_failed += 1
    
    # Test 5: Complex nested parameters
    print_separator()
    print("TEST 5: Complex nested parameters")
    print_separator()
    print("Testing complex nested parameter structures...")
    
    cache.flush_cache()
    
    params1 = {"filter": {"age": 25, "city": "NYC"}, "limit": 10}
    params2 = {"filter": {"age": 30, "city": "LA"}, "limit": 10}
    
    result1 = {"data": "NYC_users_age_25", "count": 15}
    result2 = {"data": "LA_users_age_30", "count": 22}
    
    cache.set_cache("search_users", params1, result1)
    cache.set_cache("search_users", params2, result2)
    
    print(f"Cached with params1: {result1}")
    print(f"Cached with params2: {result2}")
    
    retrieved1 = cache.get_cache("search_users", params1)
    retrieved2 = cache.get_cache("search_users", params2)
    
    print(f"\nRetrieved with params1: {retrieved1}")
    print(f"Retrieved with params2: {retrieved2}")
    
    if retrieved1 == result1 and retrieved2 == result2:
        print("\n✓ PASS: Complex nested parameters handled correctly")
        tests_passed += 1
    else:
        print("\n✗ FAIL: Complex parameters caused cache collision!")
        tests_failed += 1
    
    # Test 6: Empty parameters vs no parameters
    print_separator()
    print("TEST 6: Empty vs no parameters")
    print_separator()
    print("Testing empty dict vs None parameters...")
    
    cache.flush_cache()
    
    result_empty = {"data": "all_users", "count": 1000}
    result_none = {"data": "system_status", "status": "ok"}
    
    cache.set_cache("get_all_users", {}, result_empty)
    cache.set_cache("get_all_users", None, result_none)
    
    print(f"Cached with empty dict: {result_empty}")
    print(f"Cached with None: {result_none}")
    
    retrieved_empty = cache.get_cache("get_all_users", {})
    retrieved_none = cache.get_cache("get_all_users", None)
    
    print(f"\nRetrieved with empty dict: {retrieved_empty}")
    print(f"Retrieved with None: {retrieved_none}")
    
    if retrieved_empty == result_empty and retrieved_none == result_none:
        print("\n✓ PASS: Empty dict and None parameters handled separately")
        tests_passed += 1
    else:
        print("\n✗ FAIL: Empty dict and None parameters collided!")
        tests_failed += 1
    
    # Final summary
    print_separator()
    print("FINAL SUMMARY")
    print_separator()
    total_tests = tests_passed + tests_failed
    print(f"Total tests run: {total_tests}")
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n✓ ALL TESTS PASSED! Cache service is working correctly.")
        return 0
    else:
        print(f"\n✗ {tests_failed} TEST(S) FAILED! Cache collision bug detected.")
        print("\nThe cache key generation logic needs to be fixed to ensure:")
        print("  1. Different queries generate different cache keys")
        print("  2. Same queries (with params in any order) generate same cache keys")
        print("  3. Query type and parameters are both considered in key generation")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)