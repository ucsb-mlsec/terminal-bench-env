#!/usr/bin/env python3

import sys
import time
import tracemalloc
import random
from config_system import ConfigVersionControl

def test_correctness():
    """Test correctness of version control system"""
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    
    random.seed(42)
    config = ConfigVersionControl()
    
    # Set 100 initial configuration values
    print("\n1. Setting 100 initial configuration values...")
    for i in range(100):
        config.set(f'key_{i}', f'initial_{i}')
    
    # Commit to create version 0
    config.commit()
    print("   Created version 0")
    
    # Track what each version should contain
    expected_states = []
    current_state = {f'key_{i}': f'initial_{i}' for i in range(100)}
    expected_states.append(dict(current_state))
    
    # Create 50 versions, each modifying 5 random keys
    print("\n2. Creating 50 versions with random modifications...")
    for version in range(1, 51):
        keys_to_modify = random.sample(range(100), 5)
        for key_num in keys_to_modify:
            value = f'v{version}_key_{key_num}'
            config.set(f'key_{key_num}', value)
            current_state[f'key_{key_num}'] = value
        config.commit()
        expected_states.append(dict(current_state))
        if version % 10 == 0:
            print(f"   Created version {version}")
    
    print("\n3. Verifying individual key queries across versions...")
    # Test 20 random queries
    test_passed = True
    for _ in range(20):
        version = random.randint(0, 50)
        key_num = random.randint(0, 99)
        key = f'key_{key_num}'
        
        expected_value = expected_states[version][key]
        actual_value = config.get(key, version=version)
        
        if expected_value != actual_value:
            print(f"   FAIL: Version {version}, Key {key}")
            print(f"         Expected: {expected_value}, Got: {actual_value}")
            test_passed = False
        else:
            print(f"   OK: Version {version}, {key} = {actual_value}")
    
    if not test_passed:
        return False
    
    print("\n4. Verifying get_all() for complete configurations...")
    # Test get_all at various versions
    test_versions = [0, 10, 25, 50]
    for version in test_versions:
        expected = expected_states[version]
        actual = config.get_all(version=version)
        
        if expected != actual:
            print(f"   FAIL: Version {version} get_all() mismatch")
            print(f"         Expected {len(expected)} keys, got {len(actual)} keys")
            # Find differences
            for key in expected:
                if key not in actual:
                    print(f"         Missing key: {key}")
                elif expected[key] != actual[key]:
                    print(f"         Key {key}: expected {expected[key]}, got {actual[key]}")
            return False
        else:
            print(f"   OK: Version {version} has correct configuration ({len(actual)} keys)")
    
    print("\n5. Testing version immutability (early versions unchanged)...")
    # Check that version 0 still has original values
    version_0_state = config.get_all(version=0)
    if version_0_state != expected_states[0]:
        print("   FAIL: Version 0 was modified!")
        return False
    print("   OK: Version 0 remains unchanged")
    
    # Check a middle version
    version_25_state = config.get_all(version=25)
    if version_25_state != expected_states[25]:
        print("   FAIL: Version 25 was modified!")
        return False
    print("   OK: Version 25 remains unchanged")
    
    print("\n6. Testing latest version (version=None)...")
    latest_state = config.get_all(version=None)
    if latest_state != expected_states[50]:
        print("   FAIL: Latest version doesn't match version 50")
        return False
    print("   OK: Latest version matches version 50")
    
    # Test individual get with version=None
    key = 'key_50'
    latest_value = config.get(key, version=None)
    expected_latest = expected_states[50][key]
    if latest_value != expected_latest:
        print(f"   FAIL: get('{key}', version=None) returned {latest_value}, expected {expected_latest}")
        return False
    print(f"   OK: get('{key}', version=None) = {latest_value}")
    
    print("\n7. Testing modifications after commit don't affect committed version...")
    # Get current latest version number (should be 50)
    before_mod = config.get_all(version=50)
    
    # Make modifications without committing
    config.set('key_0', 'uncommitted_value')
    
    # Version 50 should still be unchanged
    after_mod = config.get_all(version=50)
    if before_mod != after_mod:
        print("   FAIL: Uncommitted changes affected version 50")
        return False
    print("   OK: Uncommitted changes don't affect committed versions")
    
    print("\n" + "=" * 60)
    print("ALL CORRECTNESS TESTS PASSED")
    print("=" * 60)
    return True


def test_performance():
    """Test performance with large dataset"""
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTS")
    print("=" * 60)
    
    random.seed(42)
    config = ConfigVersionControl()
    
    # Initialize with 5000 settings
    print("\n1. Initializing with 5000 settings...")
    for i in range(5000):
        config.set(f'setting_{i}', f'value_{i}')
    config.commit()
    print("   Initial commit complete")
    
    # Start tracking
    tracemalloc.start()
    start_time = time.time()
    
    print("\n2. Creating 1000 versions with 10 modifications each...")
    for version in range(1, 1001):
        # Modify 10 random settings
        for _ in range(10):
            key_num = random.randint(0, 4999)
            config.set(f'setting_{key_num}', f'v{version}_{key_num}')
        config.commit()
        
        if version % 200 == 0:
            print(f"   Created {version} versions...")
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    elapsed_time = end_time - start_time
    peak_mb = peak / (1024 * 1024)
    
    print(f"\n3. Performance Results:")
    print(f"   Time: {elapsed_time:.2f} seconds")
    print(f"   Peak Memory: {peak_mb:.2f} MB")
    print(f"   Versions Created: 1000")
    
    # Check performance requirements
    time_ok = elapsed_time < 3.0
    memory_ok = peak_mb < 100.0
    
    print(f"\n4. Performance Checks:")
    print(f"   Time < 3.0s: {'PASS' if time_ok else 'FAIL'} ({elapsed_time:.2f}s)")
    print(f"   Memory < 100MB: {'PASS' if memory_ok else 'FAIL'} ({peak_mb:.2f}MB)")
    
    if time_ok and memory_ok:
        print("\n" + "=" * 60)
        print("ALL PERFORMANCE TESTS PASSED")
        print("=" * 60)
        return True, elapsed_time, peak_mb
    else:
        print("\n" + "=" * 60)
        print("PERFORMANCE TESTS FAILED")
        print("=" * 60)
        return False, elapsed_time, peak_mb


def main():
    print("Configuration Version Control - Validation Suite\n")
    
    try:
        # Run correctness tests
        correctness_passed = test_correctness()
        
        if not correctness_passed:
            print("\n" + "=" * 60)
            print("FINAL RESULT: FAIL")
            print("Reason: Correctness tests failed")
            print("=" * 60)
            sys.exit(1)
        
        # Run performance tests
        performance_passed, elapsed_time, peak_mb = test_performance()
        
        if not performance_passed:
            print("\n" + "=" * 60)
            print("FINAL RESULT: FAIL")
            print("Reason: Performance requirements not met")
            print("=" * 60)
            sys.exit(1)
        
        # All tests passed
        print("\n" + "=" * 60)
        print("FINAL RESULT: PASS")
        print("=" * 60)
        print(f"Time: {elapsed_time:.2f}s, Memory: {peak_mb:.2f}MB")
        sys.exit(0)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("FINAL RESULT: FAIL")
        print(f"Reason: Exception occurred: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()