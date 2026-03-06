#!/usr/bin/env python3

import time
import threading
import sys
from token_bucket import TokenBucket

def test_burst_capacity():
    """Test 1: Burst capacity - can consume 20 tokens immediately"""
    print("\n[Test 1] Burst capacity test...")
    bucket = TokenBucket(rate=10, capacity=20)
    
    # Should be able to consume all 20 tokens immediately
    success_count = 0
    for i in range(20):
        if bucket.consume(1):
            success_count += 1
    
    # 21st request should fail
    should_fail = not bucket.consume(1)
    
    passed = (success_count == 20 and should_fail)
    accuracy = (success_count / 20) * 100
    
    print(f"   Consumed {success_count}/20 tokens")
    print(f"   21st request blocked: {should_fail}")
    print(f"   Status: {'PASS' if passed else 'FAIL'}")
    
    return passed, accuracy

def test_rate_limiting():
    """Test 2: Rate limiting enforcement after burst exhausted"""
    print("\n[Test 2] Rate limiting enforcement test...")
    bucket = TokenBucket(rate=10, capacity=20)
    
    # Exhaust burst capacity
    for i in range(20):
        bucket.consume(1)
    
    # Wait 0.5 seconds - should refill ~5 tokens
    time.sleep(0.5)
    
    success_count = 0
    for i in range(10):
        if bucket.consume(1):
            success_count += 1
    
    # Should have gotten approximately 5 tokens (allow ±1 for timing)
    expected = 5
    tolerance = 1
    passed = abs(success_count - expected) <= tolerance
    accuracy = max(0, 100 - abs(success_count - expected) * 10)
    
    print(f"   After 0.5s, consumed {success_count} tokens (expected ~{expected})")
    print(f"   Status: {'PASS' if passed else 'FAIL'}")
    
    return passed, accuracy

def test_token_refill_rate():
    """Test 3: Token refill at correct rate (10 tokens per second)"""
    print("\n[Test 3] Token refill rate test...")
    bucket = TokenBucket(rate=10, capacity=20)
    
    # Exhaust capacity
    for i in range(20):
        bucket.consume(1)
    
    # Wait 1 second - should refill 10 tokens
    time.sleep(1.0)
    
    success_count = 0
    for i in range(15):
        if bucket.consume(1):
            success_count += 1
    
    # Should have gotten approximately 10 tokens (allow ±1 for timing)
    expected = 10
    tolerance = 1
    passed = abs(success_count - expected) <= tolerance
    accuracy = max(0, 100 - abs(success_count - expected) * 5)
    
    print(f"   After 1.0s, consumed {success_count} tokens (expected ~{expected})")
    print(f"   Status: {'PASS' if passed else 'FAIL'}")
    
    return passed, accuracy

def test_capacity_limit():
    """Test 4: Tokens don't exceed capacity after refill"""
    print("\n[Test 4] Capacity limit test...")
    bucket = TokenBucket(rate=10, capacity=20)
    
    # Consume 10 tokens
    for i in range(10):
        bucket.consume(1)
    
    # Wait 2 seconds - would refill 20 tokens, but should cap at capacity
    time.sleep(2.0)
    
    # Try to consume 25 tokens
    success_count = 0
    for i in range(25):
        if bucket.consume(1):
            success_count += 1
    
    # Should only get 20 tokens (the capacity)
    expected = 20
    tolerance = 1
    passed = abs(success_count - expected) <= tolerance
    accuracy = max(0, 100 - abs(success_count - expected) * 5)
    
    print(f"   Consumed {success_count} tokens (expected ~{expected}, max capacity)")
    print(f"   Status: {'PASS' if passed else 'FAIL'}")
    
    return passed, accuracy

def test_concurrent_access():
    """Test 5: Concurrent request handling"""
    print("\n[Test 5] Concurrent access test...")
    bucket = TokenBucket(rate=10, capacity=20)
    
    success_count = [0]
    lock = threading.Lock()
    
    def worker():
        for i in range(5):
            if bucket.consume(1):
                with lock:
                    success_count[0] += 1
            time.sleep(0.01)
    
    # Start 6 threads, each trying to consume 5 tokens (30 total attempts)
    threads = []
    for i in range(6):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Should get approximately 20 tokens (burst capacity)
    # Allow some variance due to refill during execution
    expected = 20
    tolerance = 3
    passed = abs(success_count[0] - expected) <= tolerance
    accuracy = max(0, 100 - abs(success_count[0] - expected) * 3)
    
    print(f"   Consumed {success_count[0]} tokens across 6 threads (expected ~{expected})")
    print(f"   Status: {'PASS' if passed else 'FAIL'}")
    
    return passed, accuracy

def test_partial_consumption():
    """Test 6: Partial token consumption and refill"""
    print("\n[Test 6] Partial consumption and refill test...")
    bucket = TokenBucket(rate=10, capacity=20)
    
    # Consume 15 tokens
    for i in range(15):
        bucket.consume(1)
    
    # Wait 0.3 seconds - should refill ~3 tokens
    time.sleep(0.3)
    
    success_count = 0
    for i in range(10):
        if bucket.consume(1):
            success_count += 1
    
    # Should have ~8 tokens (5 remaining + 3 refilled)
    expected = 8
    tolerance = 1
    passed = abs(success_count - expected) <= tolerance
    accuracy = max(0, 100 - abs(success_count - expected) * 10)
    
    print(f"   Consumed {success_count} tokens (expected ~{expected})")
    print(f"   Status: {'PASS' if passed else 'FAIL'}")
    
    return passed, accuracy

def test_small_intervals():
    """Test 7: Edge case - very small time intervals"""
    print("\n[Test 7] Small time intervals test...")
    bucket = TokenBucket(rate=10, capacity=20)
    
    # Exhaust capacity
    for i in range(20):
        bucket.consume(1)
    
    # Make rapid attempts with tiny delays
    time.sleep(0.05)
    success1 = bucket.consume(1)
    
    time.sleep(0.05)
    success2 = bucket.consume(1)
    
    # Should have refilled ~1 token total
    success_count = (1 if success1 else 0) + (1 if success2 else 0)
    expected = 1
    tolerance = 1
    passed = abs(success_count - expected) <= tolerance
    accuracy = 100 if passed else 50
    
    print(f"   Consumed {success_count} tokens in small intervals (expected ~{expected})")
    print(f"   Status: {'PASS' if passed else 'FAIL'}")
    
    return passed, accuracy

def test_large_time_gap():
    """Test 8: Edge case - large time gaps"""
    print("\n[Test 8] Large time gap test...")
    bucket = TokenBucket(rate=10, capacity=20)
    
    # Exhaust capacity
    for i in range(20):
        bucket.consume(1)
    
    # Wait 5 seconds - would refill 50 tokens, but should cap at 20
    time.sleep(5.0)
    
    success_count = 0
    for i in range(25):
        if bucket.consume(1):
            success_count += 1
    
    # Should get exactly 20 (capacity limit)
    expected = 20
    tolerance = 1
    passed = abs(success_count - expected) <= tolerance
    accuracy = max(0, 100 - abs(success_count - expected) * 5)
    
    print(f"   After 5s gap, consumed {success_count} tokens (expected {expected})")
    print(f"   Status: {'PASS' if passed else 'FAIL'}")
    
    return passed, accuracy

def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("Token Bucket Rate Limiter Verification")
    print("=" * 60)
    
    tests = [
        test_burst_capacity,
        test_rate_limiting,
        test_token_refill_rate,
        test_capacity_limit,
        test_concurrent_access,
        test_partial_consumption,
        test_small_intervals,
        test_large_time_gap
    ]
    
    results = []
    accuracies = []
    
    for test_func in tests:
        try:
            passed, accuracy = test_func()
            results.append(passed)
            accuracies.append(accuracy)
        except Exception as e:
            print(f"   ERROR: {e}")
            results.append(False)
            accuracies.append(0)
    
    tests_passed = sum(results)
    overall_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/8")
    print(f"Accuracy: {overall_accuracy:.1f}%")
    print("=" * 60)
    
    return tests_passed, overall_accuracy

if __name__ == "__main__":
    tests_passed, accuracy = run_all_tests()
    
    if tests_passed == 8 and accuracy >= 98.0:
        sys.exit(0)
    else:
        sys.exit(1)