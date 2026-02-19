// API Gateway health check test
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  vus: 20,
  duration: '20s',
};

export default function () {
  const res = http.get('http://api-gateway/health');
  
  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}