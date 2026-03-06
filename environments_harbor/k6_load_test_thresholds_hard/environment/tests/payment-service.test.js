import http from 'k6/http';
import { check, sleep } from 'k6';

// Payment Service Load Test
// This script tests the payment processing API endpoint
// to validate service performance under load

export const options = {
  vus: 5,
  duration: '60s',
};

export default function () {
  // Payment processing endpoint
  const url = 'http://payment-service/api/process';
  
  // Payment request payload
  const payload = JSON.stringify({
    amount: 100,
    currency: 'USD'
  });
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  // Make POST request to process payment
  const response = http.post(url, payload, params);
  
  // Verify response status
  check(response, {
    'payment processed successfully': (r) => r.status === 200 || r.status === 201,
  });
  
  // Wait before next iteration
  sleep(2);
}