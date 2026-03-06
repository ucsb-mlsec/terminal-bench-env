#!/usr/bin/env python3

import requests
import json
import time
import uuid
import threading
import sys
import unittest

API_BASE_URL = 'http://localhost:5000'

class TestIdempotency(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Wait for server to be ready"""
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f'{API_BASE_URL}/health', timeout=1)
                if response.status_code == 200:
                    print("Server is ready")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        print("Warning: Server may not be ready")
    
    def test_same_idempotency_key_returns_same_response(self):
        """Send two requests with same idempotency key, verify identical responses and only one transaction created"""
        print("\nTest: Same idempotency key returns same response")
        
        idempotency_key = str(uuid.uuid4())
        amount = 10.00
        customer_id = f"cust_{uuid.uuid4()}"
        
        # First request
        response1 = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': amount, 'customer_id': customer_id},
            headers={'Idempotency-Key': idempotency_key}
        )
        
        self.assertEqual(response1.status_code, 200, "First request should succeed")
        data1 = response1.json()
        self.assertIn('transaction_id', data1, "Response should contain transaction_id")
        
        # Second request with same key
        response2 = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': amount, 'customer_id': customer_id},
            headers={'Idempotency-Key': idempotency_key}
        )
        
        self.assertEqual(response2.status_code, 200, "Second request should succeed")
        data2 = response2.json()
        
        # Verify responses are identical
        self.assertEqual(data1, data2, "Responses should be identical for same idempotency key")
        self.assertEqual(data1['transaction_id'], data2['transaction_id'], 
                        "Transaction IDs should match for same idempotency key")
        
        print("✓ PASS: Same idempotency key returns same response")
    
    def test_different_idempotency_keys_create_different_transactions(self):
        """Send requests with different keys, verify separate transactions"""
        print("\nTest: Different idempotency keys create different transactions")
        
        amount = 50.99
        customer_id = f"cust_{uuid.uuid4()}"
        
        # First request
        key1 = str(uuid.uuid4())
        response1 = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': amount, 'customer_id': customer_id},
            headers={'Idempotency-Key': key1}
        )
        
        self.assertEqual(response1.status_code, 200)
        data1 = response1.json()
        
        # Second request with different key
        key2 = str(uuid.uuid4())
        response2 = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': amount, 'customer_id': customer_id},
            headers={'Idempotency-Key': key2}
        )
        
        self.assertEqual(response2.status_code, 200)
        data2 = response2.json()
        
        # Verify different transaction IDs
        self.assertNotEqual(data1['transaction_id'], data2['transaction_id'],
                           "Different idempotency keys should create different transactions")
        
        print("✓ PASS: Different idempotency keys create different transactions")
    
    def test_no_idempotency_key_allows_duplicates(self):
        """Verify requests without keys are processed independently"""
        print("\nTest: No idempotency key allows duplicates")
        
        amount = 100.00
        customer_id = f"cust_{uuid.uuid4()}"
        
        # First request without key
        response1 = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': amount, 'customer_id': customer_id}
        )
        
        self.assertEqual(response1.status_code, 200)
        data1 = response1.json()
        
        # Second request without key
        response2 = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': amount, 'customer_id': customer_id}
        )
        
        self.assertEqual(response2.status_code, 200)
        data2 = response2.json()
        
        # Verify different transaction IDs
        self.assertNotEqual(data1['transaction_id'], data2['transaction_id'],
                           "Requests without idempotency keys should create separate transactions")
        
        print("✓ PASS: No idempotency key allows duplicates")
    
    def test_concurrent_requests_with_same_key(self):
        """Use threading to send concurrent requests with same idempotency key"""
        print("\nTest: Concurrent requests with same key")
        
        idempotency_key = str(uuid.uuid4())
        amount = 75.50
        customer_id = f"cust_{uuid.uuid4()}"
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = requests.post(
                    f'{API_BASE_URL}/charge',
                    json={'amount': amount, 'customer_id': customer_id},
                    headers={'Idempotency-Key': idempotency_key}
                )
                results.append(response.json())
            except Exception as e:
                errors.append(str(e))
        
        # Create 5 concurrent threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        self.assertEqual(len(errors), 0, f"No errors should occur: {errors}")
        self.assertEqual(len(results), 5, "All 5 requests should complete")
        
        # Verify all responses have the same transaction_id
        transaction_ids = [r.get('transaction_id') for r in results]
        unique_ids = set(transaction_ids)
        self.assertEqual(len(unique_ids), 1, 
                        f"Only one transaction should be created, but got: {unique_ids}")
        
        print("✓ PASS: Concurrent requests with same key handled safely")
    
    def test_invalid_idempotency_key_format(self):
        """Test with empty string, very long string, special characters"""
        print("\nTest: Invalid idempotency key format")
        
        amount = 25.00
        customer_id = f"cust_{uuid.uuid4()}"
        
        # Test empty string
        response = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': amount, 'customer_id': customer_id},
            headers={'Idempotency-Key': ''}
        )
        self.assertEqual(response.status_code, 400, "Empty idempotency key should be rejected")
        
        # Test very long string (>256 chars)
        long_key = 'x' * 300
        response = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': amount, 'customer_id': customer_id},
            headers={'Idempotency-Key': long_key}
        )
        self.assertEqual(response.status_code, 400, "Too long idempotency key should be rejected")
        
        # Test with only whitespace
        response = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': amount, 'customer_id': customer_id},
            headers={'Idempotency-Key': '   '}
        )
        self.assertEqual(response.status_code, 400, "Whitespace-only key should be rejected")
        
        print("✓ PASS: Invalid idempotency key formats rejected")
    
    def test_idempotency_key_persists_across_time(self):
        """Send request, wait, retry with same key, verify cached response"""
        print("\nTest: Idempotency key persists across time")
        
        idempotency_key = str(uuid.uuid4())
        amount = 33.33
        customer_id = f"cust_{uuid.uuid4()}"
        
        # First request
        response1 = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': amount, 'customer_id': customer_id},
            headers={'Idempotency-Key': idempotency_key}
        )
        
        self.assertEqual(response1.status_code, 200)
        data1 = response1.json()
        
        # Wait 2 seconds
        time.sleep(2)
        
        # Second request after delay
        start_time = time.time()
        response2 = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': amount, 'customer_id': customer_id},
            headers={'Idempotency-Key': idempotency_key}
        )
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        self.assertEqual(response2.status_code, 200)
        data2 = response2.json()
        
        # Verify same response
        self.assertEqual(data1, data2, "Cached response should be returned after delay")
        
        # Verify response was fast (cached)
        self.assertLess(elapsed_time, 100, 
                       f"Cached response should be fast (<100ms), but took {elapsed_time:.2f}ms")
        
        print(f"✓ PASS: Idempotency key persists (cached response in {elapsed_time:.2f}ms)")
    
    def test_different_parameters_same_key(self):
        """Send request, retry with same key but different parameters"""
        print("\nTest: Different parameters with same key returns original response")
        
        idempotency_key = str(uuid.uuid4())
        customer_id = f"cust_{uuid.uuid4()}"
        
        # First request
        response1 = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': 10.00, 'customer_id': customer_id},
            headers={'Idempotency-Key': idempotency_key}
        )
        
        self.assertEqual(response1.status_code, 200)
        data1 = response1.json()
        
        # Second request with different amount but same key
        response2 = requests.post(
            f'{API_BASE_URL}/charge',
            json={'amount': 99.99, 'customer_id': customer_id},
            headers={'Idempotency-Key': idempotency_key}
        )
        
        self.assertEqual(response2.status_code, 200)
        data2 = response2.json()
        
        # Verify original response is returned
        self.assertEqual(data1, data2, 
                        "Original response should be returned even with different parameters")
        self.assertEqual(data1['transaction_id'], data2['transaction_id'],
                        "Transaction ID should match original")
        
        print("✓ PASS: Same key with different parameters returns original response")

def main():
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestIdempotency)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("ALL TESTS PASSED")
        print("="*70)
        return 0
    else:
        print("SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print("="*70)
        return 1

if __name__ == '__main__':
    sys.exit(main())