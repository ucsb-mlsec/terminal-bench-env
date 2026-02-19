import http from 'k6/http';
import { sleep } from 'k6';

// Monitoring endpoints performance test

export const options = {
  stages: [
    { duration: '10s', target: 5 },
    { duration: '20s', target: 10 }
  ]
};

export default function () {
  http.get('http://monitoring/metrics');
  http.get('http://monitoring/health');
  sleep(1);
}