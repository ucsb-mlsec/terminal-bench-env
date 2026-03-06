#!/usr/bin/env python3

import json
from circuit_breaker import CircuitBreaker, CircuitBreakerOpenException


class PaymentClient:
    def __init__(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config['failure_threshold'],
            timeout_seconds=config['timeout_seconds']
        )
    
    def process_payment(self, amount, card_number):
        try:
            result = self.circuit_breaker.call(
                lambda: self._make_payment_request(amount, card_number)
            )
            return result
        except CircuitBreakerOpenException:
            return {
                'status': 'circuit_open',
                'message': 'Payment service unavailable'
            }
    
    def _make_payment_request(self, amount, card_number):
        # This simulates calling the actual payment service
        # In tests, this will be mocked to simulate failures
        return {
            'status': 'success',
            'transaction_id': 'txn_12345',
            'amount': amount
        }