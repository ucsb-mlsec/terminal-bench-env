#!/usr/bin/env python3

import sys
import time
from payment_client import PaymentClient


class TestablePaymentClient(PaymentClient):
    """Extended PaymentClient that allows simulation of service failures for testing."""
    
    def __init__(self):
        super().__init__()
        self._should_fail = False
    
    def set_service_status(self, should_fail):
        """Control whether the payment service should fail or succeed."""
        self._should_fail = should_fail
    
    def _make_payment_request(self, amount):
        """Override to simulate service behavior based on test configuration."""
        if self._should_fail:
            raise Exception('Payment service unavailable')
        return {'status': 'success', 'amount': amount}


def test_circuit_closed_normal_operation():
    """Test that circuit breaker allows requests through when service is healthy."""
    print("\n[TEST] Circuit Closed - Normal Operation")
    
    try:
        client = TestablePaymentClient()
        client.set_service_status(False)  # Service is healthy
        
        # Make several successful calls
        for i in range(5):
            result = client.make_payment(100.0)
            assert result is not None, f"Payment {i+1} should succeed"
            assert result.get('status') == 'success', f"Payment {i+1} should have success status"
        
        print("  ✓ All payments succeeded with healthy service")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_circuit_opens_after_threshold():
    """Test that circuit opens after reaching failure threshold."""
    print("\n[TEST] Circuit Opens After Threshold")
    
    try:
        client = TestablePaymentClient()
        client.set_service_status(True)  # Service is failing
        
        # Get the failure threshold from config
        threshold = client.circuit_breaker.failure_threshold
        
        # Make enough failed calls to reach threshold
        failure_count = 0
        for i in range(threshold):
            result = client.make_payment(100.0)
            if result is None:
                failure_count += 1
        
        assert failure_count == threshold, f"Expected {threshold} failures, got {failure_count}"
        print(f"  ✓ Reached failure threshold ({threshold} failures)")
        
        # Circuit should now be open - next call should be rejected immediately
        # without even calling the service
        start_time = time.time()
        result = client.make_payment(100.0)
        elapsed = time.time() - start_time
        
        assert result is None, "Payment should fail when circuit is open"
        assert elapsed < 0.1, "Circuit breaker should reject immediately without calling service"
        print("  ✓ Circuit opened and rejects requests immediately")
        
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_circuit_remains_open_during_timeout():
    """Test that circuit stays open and rejects calls during timeout period."""
    print("\n[TEST] Circuit Remains Open During Timeout")
    
    try:
        client = TestablePaymentClient()
        client.set_service_status(True)  # Service is failing
        
        threshold = client.circuit_breaker.failure_threshold
        
        # Trigger circuit to open
        for i in range(threshold):
            client.make_payment(100.0)
        
        # Make one more call to confirm it's open
        result = client.make_payment(100.0)
        assert result is None, "Circuit should be open"
        print("  ✓ Circuit is open")
        
        # Wait for half the timeout period
        timeout = client.circuit_breaker.timeout_seconds
        wait_time = timeout / 2
        print(f"  ⏱ Waiting {wait_time}s (half of {timeout}s timeout)...")
        time.sleep(wait_time)
        
        # Circuit should still be open
        result = client.make_payment(100.0)
        assert result is None, "Circuit should still be open during timeout period"
        print("  ✓ Circuit remains open during timeout period")
        
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_circuit_half_open_recovery():
    """Test that circuit enters half-open state after timeout and closes on success."""
    print("\n[TEST] Circuit Half-Open Recovery")
    
    try:
        client = TestablePaymentClient()
        client.set_service_status(True)  # Service is failing
        
        threshold = client.circuit_breaker.failure_threshold
        
        # Trigger circuit to open
        for i in range(threshold):
            client.make_payment(100.0)
        
        result = client.make_payment(100.0)
        assert result is None, "Circuit should be open"
        print("  ✓ Circuit opened after failures")
        
        # Wait for timeout period to elapse
        timeout = client.circuit_breaker.timeout_seconds
        print(f"  ⏱ Waiting {timeout + 0.5}s for timeout to elapse...")
        time.sleep(timeout + 0.5)
        
        # Service has now recovered
        client.set_service_status(False)
        
        # Next call should attempt to go through (half-open state)
        result = client.make_payment(100.0)
        assert result is not None, "Payment should succeed in half-open state"
        assert result.get('status') == 'success', "Payment should succeed with recovered service"
        print("  ✓ Circuit entered half-open state and test request succeeded")
        
        # Circuit should now be closed - subsequent calls should work
        result = client.make_payment(100.0)
        assert result is not None, "Payment should succeed after circuit closes"
        print("  ✓ Circuit closed after successful recovery")
        
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_circuit_reopens_on_half_open_failure():
    """Test that circuit reopens if test call fails in half-open state."""
    print("\n[TEST] Circuit Reopens on Half-Open Failure")
    
    try:
        client = TestablePaymentClient()
        client.set_service_status(True)  # Service is failing
        
        threshold = client.circuit_breaker.failure_threshold
        
        # Trigger circuit to open
        for i in range(threshold):
            client.make_payment(100.0)
        
        result = client.make_payment(100.0)
        assert result is None, "Circuit should be open"
        print("  ✓ Circuit opened after failures")
        
        # Wait for timeout period
        timeout = client.circuit_breaker.timeout_seconds
        print(f"  ⏱ Waiting {timeout + 0.5}s for timeout to elapse...")
        time.sleep(timeout + 0.5)
        
        # Service is still failing
        result = client.make_payment(100.0)
        assert result is None, "Payment should fail when service still down"
        print("  ✓ Test request failed in half-open state")
        
        # Circuit should reopen - next call should be rejected immediately
        start_time = time.time()
        result = client.make_payment(100.0)
        elapsed = time.time() - start_time
        
        assert result is None, "Payment should fail when circuit reopens"
        assert elapsed < 0.1, "Circuit should reject immediately after reopening"
        print("  ✓ Circuit reopened and rejects requests immediately")
        
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("CIRCUIT BREAKER TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_circuit_closed_normal_operation,
        test_circuit_opens_after_threshold,
        test_circuit_remains_open_during_timeout,
        test_circuit_half_open_recovery,
        test_circuit_reopens_on_half_open_failure
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        if test_func():
            passed += 1
            print(f"  → PASS")
        else:
            failed += 1
            print(f"  → FAIL")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())