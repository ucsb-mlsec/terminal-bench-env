import http from 'k6/http';
import { check, sleep } from 'k6';

// User Service Load Test
// Tests the user-service API endpoints for performance

export const options = {
  vus: 10,
  duration: '30s',
};

export default function () {
  // Make GET request to user service endpoint
  const response = http.get('http://user-service/api/users');
  
  // Verify response status
  check(response, {
    'status is 200': (r) => r.status === 200,
  });
  
  // Wait between iterations
  sleep(1);
}