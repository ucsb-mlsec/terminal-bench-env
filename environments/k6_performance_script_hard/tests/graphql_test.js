import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
    vus: 10,
    duration: '30s',
    thresholds: {
        'http_req_durations': ['p(95)<500'],
        'http_req_failed': ['rate<0.01'],
        'response_time': ['avg<200']
    }
};

export default function() {
    const query = `
        query {
            products {
                id
                name
                price
            }
        }
    `;

    const payload = JSON.stringify({
        query: query
    });

    const params = {
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    };

    const res = http.post('http://localhost:4000/graphql', payload, params);

    check(res, {
        'status is 200': (r) => r.status === 200,
        'has data': (r) => r.body.includes('products')
    });

    sleep(1);
}