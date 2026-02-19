#!/usr/bin/env python3

import time
from rate_limiter import RateLimiter

def test_rate_limiter():
    """Test the RateLimiter implementation with various scenarios."""
    
    print("=" * 60)
    print("Testing RateLimiter Implementation")
    print("=" * 60)
    
    # Create rate limiter instance
    limiter = RateLimiter(capacity=10, refill_rate=2)
    all_tests_passed = True
    
    # Test 1: Make 10 rapid requests from 'client1' - all should succeed
    print("\nTest 1: Making 10 rapid requests from 'client1'")
    print("-" * 60)
    test1_passed = True
    for i in range(10):
        result = limiter.allow_request('client1')
        if not result:
            print(f"  Request {i+1}: FAILED (expected True, got False)")
            test1_passed = False
            all_tests_passed = False
        else:
            print(f"  Request {i+1}: PASSED (allowed)")
    
    if test1_passed:
        print("✓ Test 1 PASSED: All 10 requests were allowed")
    else:
        print("✗ Test 1 FAILED: Some requests were incorrectly denied")
    
    # Test 2: Make an 11th rapid request from 'client1' - should fail
    print("\nTest 2: Making 11th rapid request from 'client1'")
    print("-" * 60)
    result = limiter.allow_request('client1')
    if not result:
        print(f"  Request 11: PASSED (correctly denied)")
        print("✓ Test 2 PASSED: 11th request was correctly denied")
    else:
        print(f"  Request 11: FAILED (expected False, got True)")
        print("✗ Test 2 FAILED: 11th request should have been denied")
        all_tests_passed = False
    
    # Test 3: Wait 1 second, make 2 requests from 'client1' - should succeed
    print("\nTest 3: Waiting 1 second and making 2 requests from 'client1'")
    print("-" * 60)
    print("  Waiting 1 second for token refill...")
    time.sleep(1)
    test3_passed = True
    for i in range(2):
        result = limiter.allow_request('client1')
        if not result:
            print(f"  Request {i+1} after wait: FAILED (expected True, got False)")
            test3_passed = False
            all_tests_passed = False
        else:
            print(f"  Request {i+1} after wait: PASSED (allowed)")
    
    if test3_passed:
        print("✓ Test 3 PASSED: Tokens refilled correctly, 2 requests allowed")
    else:
        print("✗ Test 3 FAILED: Tokens did not refill correctly")
    
    # Test 4: Make requests from 'client2' - should succeed independently
    print("\nTest 4: Making requests from 'client2' (independent tracking)")
    print("-" * 60)
    test4_passed = True
    for i in range(5):
        result = limiter.allow_request('client2')
        if not result:
            print(f"  Request {i+1}: FAILED (expected True, got False)")
            test4_passed = False
            all_tests_passed = False
        else:
            print(f"  Request {i+1}: PASSED (allowed)")
    
    if test4_passed:
        print("✓ Test 4 PASSED: Client2 tracked independently")
    else:
        print("✗ Test 4 FAILED: Clients not tracked independently")
    
    # Test 5: Test slow requests (one per second) from 'client3' - should never be denied
    print("\nTest 5: Making slow requests (1 per second) from 'client3'")
    print("-" * 60)
    test5_passed = True
    for i in range(5):
        result = limiter.allow_request('client3')
        if not result:
            print(f"  Request {i+1}: FAILED (expected True, got False)")
            test5_passed = False
            all_tests_passed = False
        else:
            print(f"  Request {i+1}: PASSED (allowed)")
        if i < 4:  # Don't sleep after the last request
            time.sleep(1)
    
    if test5_passed:
        print("✓ Test 5 PASSED: Slow requests never denied")
    else:
        print("✗ Test 5 FAILED: Slow requests should not be denied")
    
    # Test 6: Verify get_tokens method works
    print("\nTest 6: Testing get_tokens method")
    print("-" * 60)
    try:
        tokens_client1 = limiter.get_tokens('client1')
        tokens_client2 = limiter.get_tokens('client2')
        tokens_client3 = limiter.get_tokens('client3')
        print(f"  client1 tokens: {tokens_client1:.2f}")
        print(f"  client2 tokens: {tokens_client2:.2f}")
        print(f"  client3 tokens: {tokens_client3:.2f}")
        print("✓ Test 6 PASSED: get_tokens method works")
    except Exception as e:
        print(f"✗ Test 6 FAILED: get_tokens method error - {e}")
        all_tests_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 60)
        return True
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("=" * 60)
        return False

if __name__ == "__main__":
    test_passed = test_rate_limiter()
    exit(0 if test_passed else 1)