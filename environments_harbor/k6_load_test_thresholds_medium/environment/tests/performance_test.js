import http from 'k6/http';
import { check } from 'k6';

export let options = {
  thresholds: {
    'response_time': ['p(95)<500'],
    'http_req_duration': ['p(99) < 1000ms'],
    'checks': ['rate>0.99'],
    'http_reqs': ['count>=100'],
  },
  vus: 10,
  duration: '30s',
};

export default function () {
  const response = http.get('https://test.k6.io');
  
  check(response, {
    'status is 200': (r) => r.status === 200,
  });
}