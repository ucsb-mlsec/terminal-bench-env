// K6 Load Test Configuration
// WARNING: This configuration has known issues and should be updated

export const BASE_URL = 'http://localhost:8080';
export const GRAPHQL_URL = 'http://localhost:4000';
export const WS_URL = 'ws://localhost:3000';

// Test execution parameters
export const TEST_DURATION = '30s'; // Too short for proper load testing
export const VUS = 10; // Static VUs - no ramping configured

// Performance thresholds - INCORRECT VALUES
export const THRESHOLDS = {
  http_req_duration: ['p(95)<10000'], // 10 seconds is too permissive
  http_req_failed: ['rate<0.5'], // 50% failure rate is way too high
};

// User behavior simulation
export const THINK_TIME = 0; // No think time between actions - unrealistic

// API endpoints
export const ENDPOINTS = {
  login: '/api/auth/login',
  products: '/api/products',
  cart: '/api/cart',
  checkout: '/api/checkout',
};

// Missing: Rate limiting parameters, scenario configurations, proper stage definitions