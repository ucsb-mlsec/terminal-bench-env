import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import ws from 'k6/ws';
import { SharedArray } from 'k6/data';
import { randomIntBetween, randomItem } from 'k6/execution';

// Custom metrics for business KPIs
const conversionRate = new Rate('conversion_rate');
const cartAbandonmentRate = new Rate('cart_abandonment_rate');
const checkoutDuration = new Trend('checkout_duration');
const wsConnectionDuration = new Trend('ws_connection_duration');
const apiErrorRate = new Rate('api_error_rate');
const authFailureRate = new Rate('auth_failure_rate');
const productViewCounter = new Counter('product_views');
const cartAddCounter = new Counter('cart_adds');
const orderCounter = new Counter('orders_completed');

// Test configuration
export const options = {
    scenarios: {
        // Gradual ramp-up for browsing users
        browsing_users: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                { duration: '2m', target: 20 },
                { duration: '5m', target: 50 },
                { duration: '3m', target: 80 },
                { duration: '5m', target: 80 },
                { duration: '2m', target: 0 },
            ],
            gracefulRampDown: '30s',
            exec: 'browsingUserJourney',
        },
        // Purchasing users with higher conversion intent
        purchasing_users: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                { duration: '3m', target: 10 },
                { duration: '5m', target: 25 },
                { duration: '4m', target: 40 },
                { duration: '5m', target: 40 },
                { duration: '2m', target: 0 },
            ],
            gracefulRampDown: '30s',
            exec: 'purchasingUserJourney',
        },
        // WebSocket notification service testing
        websocket_users: {
            executor: 'constant-vus',
            vus: 30,
            duration: '17m',
            exec: 'websocketUserJourney',
            startTime: '1m',
        },
        // GraphQL API stress test
        graphql_queries: {
            executor: 'constant-arrival-rate',
            rate: 50,
            timeUnit: '1s',
            duration: '15m',
            preAllocatedVUs: 20,
            maxVUs: 100,
            exec: 'graphqlQueries',
            startTime: '30s',
        },
        // Rate limiting test
        rate_limit_test: {
            executor: 'constant-arrival-rate',
            rate: 200,
            timeUnit: '1s',
            duration: '1m',
            preAllocatedVUs: 50,
            maxVUs: 200,
            exec: 'rateLimitTest',
            startTime: '10m',
        },
    },
    thresholds: {
        'http_req_duration': ['p(95)<500', 'p(99)<1000'],
        'http_req_duration{endpoint:login}': ['p(95)<300'],
        'http_req_duration{endpoint:product}': ['p(95)<200'],
        'http_req_duration{endpoint:cart}': ['p(95)<400'],
        'http_req_duration{endpoint:checkout}': ['p(95)<800'],
        'http_req_failed': ['rate<0.05'],
        'http_req_failed{endpoint:checkout}': ['rate<0.01'],
        'api_error_rate': ['rate<0.02'],
        'auth_failure_rate': ['rate<0.001'],
        'conversion_rate': ['rate>0.15'],
        'cart_abandonment_rate': ['rate<0.40'],
        'checkout_duration': ['p(95)<3000'],
        'ws_connection_duration': ['p(95)<2000'],
        'checks': ['rate>0.95'],
    },
};

// Shared test data
const products = new SharedArray('products', function() {
    return [
        { id: 101, name: 'Laptop', price: 999.99 },
        { id: 102, name: 'Smartphone', price: 699.99 },
        { id: 103, name: 'Headphones', price: 199.99 },
        { id: 104, name: 'Tablet', price: 499.99 },
        { id: 105, name: 'Smartwatch', price: 299.99 },
        { id: 106, name: 'Camera', price: 799.99 },
        { id: 107, name: 'Speaker', price: 149.99 },
        { id: 108, name: 'Monitor', price: 399.99 },
    ];
});

const BASE_URL_REST = 'http://localhost:8080';
const BASE_URL_GRAPHQL = 'http://localhost:4000';
const BASE_URL_WS = 'ws://localhost:3000';

// Helper function for authentication
function authenticate(username, password) {
    const loginUrl = `${BASE_URL_REST}/api/auth/login`;
    const payload = JSON.stringify({
        username: username,
        password: password,
    });
    
    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
        tags: { endpoint: 'login' },
    };
    
    const response = http.post(loginUrl, payload, params);
    
    const loginSuccess = check(response, {
        'login status is 200': (r) => r.status === 200,
        'login returns token': (r) => r.json('token') !== undefined,
    });
    
    authFailureRate.add(!loginSuccess);
    
    if (loginSuccess && response.json('token')) {
        return response.json('token');
    }
    
    apiErrorRate.add(1);
    return null;
}

// Helper function to create authorized headers
function getAuthHeaders(token) {
    return {
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
        },
    };
}

// Browsing user journey - views products but may not purchase
export function browsingUserJourney() {
    let token = null;
    let userConverted = false;
    let addedToCart = false;
    
    // 70% of browsing users are authenticated
    if (Math.random() < 0.7) {
        const userId = Math.floor(Math.random() * 1000);
        token = authenticate(`user${userId}`, 'testpass123');
        
        if (!token) {
            return;
        }
    }
    
    sleep(randomIntBetween(1, 3)); // Think time after login
    
    // Browse 2-5 products
    const numProductsToView = randomIntBetween(2, 5);
    for (let i = 0; i < numProductsToView; i++) {
        const product = products[Math.floor(Math.random() * products.length)];
        
        const productUrl = `${BASE_URL_REST}/api/products/${product.id}`;
        const params = token ? 
            { ...getAuthHeaders(token), tags: { endpoint: 'product' } } :
            { tags: { endpoint: 'product' } };
        
        const response = http.get(productUrl, params);
        
        check(response, {
            'product view status is 200': (r) => r.status === 200,
            'product data is valid': (r) => r.json('id') !== undefined,
        });
        
        productViewCounter.add(1);
        
        if (response.status !== 200) {
            apiErrorRate.add(1);
        }
        
        sleep(randomIntBetween(2, 5)); // Think time while viewing product
        
        // 40% chance to add to cart for browsing users
        if (token && Math.random() < 0.4) {
            const cartUrl = `${BASE_URL_REST}/api/cart/add`;
            const cartPayload = JSON.stringify({
                productId: product.id,
                quantity: 1,
            });
            
            const cartParams = {
                ...getAuthHeaders(token),
                tags: { endpoint: 'cart' },
            };
            
            const cartResponse = http.post(cartUrl, cartPayload, cartParams);
            
            const cartSuccess = check(cartResponse, {
                'add to cart status is 200 or 201': (r) => r.status === 200 || r.status === 201,
                'cart response has items': (r) => r.json('items') !== undefined,
            });
            
            if (cartSuccess) {
                cartAddCounter.add(1);
                addedToCart = true;
            } else {
                apiErrorRate.add(1);
            }
            
            sleep(randomIntBetween(1, 2));
        }
    }
    
    // Track cart abandonment
    if (addedToCart) {
        cartAbandonmentRate.add(1); // Browsing users typically abandon
    }
}

// Purchasing user journey - high intent to complete purchase
export function purchasingUserJourney() {
    const userId = Math.floor(Math.random() * 1000);
    const token = authenticate(`user${userId}`, 'testpass123');
    
    if (!token) {
        return;
    }
    
    sleep(randomIntBetween(1, 2));
    
    let cartItems = [];
    const checkoutStartTime = Date.now();
    
    // View and add 1-3 products to cart
    const numProducts = randomIntBetween(1, 3);
    for (let i = 0; i < numProducts; i++) {
        const product = products[Math.floor(Math.random() * products.length)];
        
        // View product
        const productUrl = `${BASE_URL_REST}/api/products/${product.id}`;
        const productResponse = http.get(productUrl, {
            ...getAuthHeaders(token),
            tags: { endpoint: 'product' },
        });
        
        check(productResponse, {
            'product view successful': (r) => r.status === 200,
        });
        
        productViewCounter.add(1);
        sleep(randomIntBetween(2, 4));
        
        // Add to cart
        const cartUrl = `${BASE_URL_REST}/api/cart/add`;
        const cartPayload = JSON.stringify({
            productId: product.id,
            quantity: randomIntBetween(1, 2),
        });
        
        const cartResponse = http.post(cartUrl, cartPayload, {
            ...getAuthHeaders(token),
            tags: { endpoint: 'cart' },
        });
        
        const addSuccess = check(cartResponse, {
            'product added to cart': (r) => r.status === 200 || r.status === 201,
        });
        
        if (addSuccess) {
            cartAddCounter.add(1);
            cartItems.push(product);
        } else {
            apiErrorRate.add(1);
        }
        
        sleep(randomIntBetween(1, 3));
    }
    
    if (cartItems.length === 0) {
        return;
    }
    
    // View cart
    const viewCartUrl = `${BASE_URL_REST}/api/cart`;
    const cartViewResponse = http.get(viewCartUrl, {
        ...getAuthHeaders(token),
        tags: { endpoint: 'cart' },
    });
    
    check(cartViewResponse, {
        'cart view successful': (r) => r.status === 200,
        'cart contains items': (r) => r.json('items') && r.json('items').length > 0,
    });
    
    sleep(randomIntBetween(2, 4)); // Review cart
    
    // 85% of purchasing users complete checkout
    if (Math.random() < 0.85) {
        const checkoutUrl = `${BASE_URL_REST}/api/checkout`;
        const checkoutPayload = JSON.stringify({
            paymentMethod: 'credit_card',
            shippingAddress: {
                street: '123 Test St',
                city: 'Test City',
                zipCode: '12345',
            },
        });
        
        const checkoutResponse = http.post(checkoutUrl, checkoutPayload, {
            ...getAuthHeaders(token),
            tags: { endpoint: 'checkout' },
        });
        
        const checkoutSuccess = check(checkoutResponse, {
            'checkout status is 200 or 201': (r) => r.status === 200 || r.status === 201,
            'order id received': (r) => r.json('orderId') !== undefined,
            'checkout completed': (r) => r.json('status') === 'completed',
        });
        
        const totalCheckoutTime = Date.now() - checkoutStartTime;
        checkoutDuration.add(totalCheckoutTime);
        
        if (checkoutSuccess) {
            conversionRate.add(1);
            cartAbandonmentRate.add(0);
            orderCounter.add(1);
        } else {
            conversionRate.add(0);
            cartAbandonmentRate.add(1);
            apiErrorRate.add(1);
        }
    } else {
        // User abandons cart at checkout
        conversionRate.add(0);
        cartAbandonmentRate.add(1);
    }
    
    sleep(randomIntBetween(1, 2));
}

// WebSocket notification service testing
export function websocketUserJourney() {
    const userId = Math.floor(Math.random() * 10000);
    const token = authenticate(`user${userId}`, 'testpass123');
    
    if (!token) {
        return;
    }
    
    const wsUrl = `${BASE_URL_WS}/notifications?token=${token}`;
    const startTime = Date.now();
    
    const response = ws.connect(wsUrl, {
        tags: { endpoint: 'websocket' },
    }, function(socket) {
        socket.on('open', function() {
            wsConnectionDuration.add(Date.now() - startTime);
            
            // Send heartbeat every 30 seconds
            socket.setInterval(function() {
                socket.send(JSON.stringify({ type: 'ping' }));
            }, 30000);
            
            // Subscribe to notifications
            socket.send(JSON.stringify({
                type: 'subscribe',
                channels: ['orders', 'promotions'],
            }));
        });
        
        socket.on('message', function(msg) {
            const data = JSON.parse(msg);
            check(data, {
                'ws message has type': (d) => d.type !== undefined,
                'ws message is valid': (d) => d !== null,
            });
        });
        
        socket.on('error', function(e) {
            apiErrorRate.add(1);
        });
        
        socket.on('close', function() {
            // Connection closed, cleanup handled automatically
        });
        
        // Keep connection alive for 60-120 seconds
        socket.setTimeout(function() {
            socket.close();
        }, randomIntBetween(60000, 120000));
    });
    
    check(response, {
        'ws connection established': (r) => r && r.status === 101,
    });
}

// GraphQL API queries
export function graphqlQueries() {
    const userId = Math.floor(Math.random() * 1000);
    const token = authenticate(`user${userId}`, 'testpass123');
    
    if (!token) {
        return;
    }
    
    const queries = [
        // Query user profile and order history
        {
            query: `
                query UserProfile {
                    user {
                        id
                        email
                        orders {
                            id
                            total
                            status
                            items {
                                productId
                                quantity
                            }
                        }
                    }
                }
            `,
            name: 'userProfile',
        },
        // Query product catalog with filters
        {
            query: `
                query ProductCatalog($category: String, $minPrice: Float, $maxPrice: Float) {
                    products(category: $category, minPrice: $minPrice, maxPrice: $maxPrice) {
                        id
                        name
                        price
                        rating
                        inStock
                    }
                }
            `,
            variables: {
                category: 'electronics',
                minPrice: 100,
                maxPrice: 1000,
            },
            name: 'productCatalog',
        },
        // Query recommendations
        {
            query: `
                query Recommendations {
                    recommendations {
                        id
                        name
                        price
                        reason
                    }
                }
            `,
            name: 'recommendations',
        },
    ];
    
    const query = queries[Math.floor(Math.random() * queries.length)];
    
    const payload = JSON.stringify({
        query: query.query,
        variables: query.variables || {},
    });
    
    const response = http.post(BASE_URL_GRAPHQL, payload, {
        ...getAuthHeaders(token),
        tags: { endpoint: 'graphql', query_type: query.name },
    });
    
    const success = check(response, {
        'graphql status is 200': (r) => r.status === 200,
        'graphql no errors': (r) => !r.json('errors'),
        'graphql has data': (r) => r.json('data') !== undefined,
    });
    
    if (!success) {
        apiErrorRate.add(1);
    }
    
    sleep(0.5);
}

// Rate limiting test
export function rateLimitTest() {
    const token = authenticate(`ratetest${Math.floor(Math.random() * 100)}`, 'testpass123');
    
    if (!token) {
        return;
    }
    
    const productId = products[Math.floor(Math.random() * products.length)].id;
    const response = http.get(`${BASE_URL_REST}/api/products/${productId}`, {
        ...getAuthHeaders(token),
        tags: { endpoint: 'rate_limit_test' },
    });
    
    check(response, {
        'response received': (r) => r.status !== 0,
        'rate limit handled': (r) => r.status === 200 || r.status === 429,
    });
    
    if (response.status === 429) {
        check(response, {
            'rate limit headers present': (r) => r.headers['X-RateLimit-Remaining'] !== undefined,
            'retry-after header present': (r) => r.headers['Retry-After'] !== undefined,
        });
    }
}

// Default function for standalone execution
export default function() {
    browsingUserJourney();
}