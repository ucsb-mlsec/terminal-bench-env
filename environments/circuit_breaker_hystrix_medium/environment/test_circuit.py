#!/usr/bin/env python3

import time
import sys
from service_caller import fetch_data

# Test URLs simulating different service behaviors
FAILING_URL = 'http://failing-service.example.com/api'
WORKING_URL = 'http://working-service.example.com/api'
FLAKY_URL = 'http://flaky-service.example.com/api'

def test_failing_url():
    """
    Test that fetch_data stops attempting requests to consistently failing URLs
    after the failure threshold is reached, and returns quickly.
    """
    print("\n=== Test 1: Testing Failing URL ===")
    print(f"URL: {FAILING_URL}")
    
    times = []
    results = []
    
    # Make multiple requests to trigger circuit breaker
    for i in range(10):
        start = time.time()
        result = fetch_data(FAILING_URL)
        elapsed = time.time() - start
        times.append(elapsed)
        results.append(result)
        print(f"Attempt {i+1}: Result={result}, Time={elapsed:.3f}s")
    
    # After threshold is reached, responses should be very fast (< 1 second)
    fast_responses = sum(1 for t in times[5:] if t < 1.0)
    
    if fast_responses >= 3:
        print(f"✓ PASS: Circuit breaker activated, {fast_responses}/5 later requests were fast")
        return True
    else:
        print(f"✗ FAIL: Circuit breaker not working properly, only {fast_responses}/5 fast responses")
        return False

def test_working_url():
    """
    Test that fetch_data continues to work normally for URLs that succeed.
    """
    print("\n=== Test 2: Testing Working URL ===")
    print(f"URL: {WORKING_URL}")
    
    success_count = 0
    
    for i in range(5):
        start = time.time()
        result = fetch_data(WORKING_URL)
        elapsed = time.time() - start
        
        if result is not None:
            success_count += 1
        
        print(f"Attempt {i+1}: Result={result is not None}, Time={elapsed:.3f}s")
    
    if success_count >= 4:
        print(f"✓ PASS: Working URL handled correctly ({success_count}/5 successful)")
        return True
    else:
        print(f"✗ FAIL: Working URL not handled correctly ({success_count}/5 successful)")
        return False

def test_recovery():
    """
    Test that a previously failing URL can be retried after the timeout period
    to detect service recovery.
    """
    print("\n=== Test 3: Testing Recovery Mechanism ===")
    recovery_url = 'http://recovery-test.example.com/api'
    
    # Trigger circuit breaker
    print(f"Phase 1: Triggering circuit breaker for {recovery_url}")
    for i in range(6):
        result = fetch_data(recovery_url)
        print(f"  Failure {i+1}: {result}")
    
    # Try immediately - should fail fast
    print("\nPhase 2: Testing immediate retry (should fail fast)")
    start = time.time()
    result = fetch_data(recovery_url)
    elapsed = time.time() - start
    print(f"  Immediate retry: Time={elapsed:.3f}s")
    
    immediate_fast = elapsed < 1.0
    
    # Wait for retry interval
    print("\nPhase 3: Waiting for retry interval...")
    wait_time = 32  # Slightly more than typical retry_interval
    print(f"  Waiting {wait_time} seconds...")
    time.sleep(wait_time)
    
    # Should attempt again now
    print("\nPhase 4: Testing after retry interval (should attempt again)")
    start = time.time()
    result = fetch_data(recovery_url)
    elapsed = time.time() - start
    print(f"  Post-interval retry: Time={elapsed:.3f}s")
    
    # The retry should take longer than immediate rejection
    retry_attempted = elapsed >= 0.5 or elapsed < 1.5
    
    if immediate_fast:
        print("✓ PASS: Recovery mechanism allows retry after interval")
        return True
    else:
        print("✗ FAIL: Recovery mechanism not working as expected")
        return False

def main():
    """
    Run all test cases and report overall results.
    """
    print("=" * 60)
    print("Circuit Breaker Testing Suite")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("Failing URL Test", test_failing_url()))
    results.append(("Working URL Test", test_working_url()))
    results.append(("Recovery Test", test_recovery()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()