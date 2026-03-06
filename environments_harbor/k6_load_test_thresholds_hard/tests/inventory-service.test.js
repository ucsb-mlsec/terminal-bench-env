import http from 'k6/http';
import { check, sleep } from 'k6';

// Load test script for inventory-service
// This test validates the inventory API endpoints

export let options = {
  vus: 15,
  duration: '45s',
};

export default function() {
  // Test the items list endpoint
  let listResponse = http.get('http://inventory-service/api/items');
  
  check(listResponse, {
    'list items status is 200': (r) => r.status === 200,
  });
  
  // Test the individual item endpoint
  let itemResponse = http.get('http://inventory-service/api/items/123');
  
  check(itemResponse, {
    'get item status is 200': (r) => r.status === 200,
  });
  
  sleep(0.5);
}