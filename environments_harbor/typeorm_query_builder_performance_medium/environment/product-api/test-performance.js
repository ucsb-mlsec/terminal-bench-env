const http = require('http');
const fs = require('fs');
const path = require('path');

let queryCount = 0;
let server;

// Mock logger to intercept TypeORM queries
const originalLog = console.log;
console.log = function(...args) {
  const message = args.join(' ');
  if (message.includes('SELECT') || message.includes('query:')) {
    if (message.toLowerCase().includes('select')) {
      queryCount++;
    }
  }
  originalLog.apply(console, args);
};

async function runTest() {
  try {
    console.log('Starting performance test...\n');

    // Start the server
    const serverModule = require('./server.js');
    server = serverModule.server || serverModule;

    // Wait for server to be ready
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Reset query count before making the request
    queryCount = 0;

    // Measure response time
    const startTime = Date.now();

    // Make HTTP request
    const responseData = await new Promise((resolve, reject) => {
      const req = http.get('http://localhost:3000/api/products', (res) => {
        let data = '';

        res.on('data', (chunk) => {
          data += chunk;
        });

        res.on('end', () => {
          if (res.statusCode === 200) {
            try {
              const parsed = JSON.parse(data);
              resolve(parsed);
            } catch (e) {
              reject(new Error('Failed to parse JSON response'));
            }
          } else {
            reject(new Error(`HTTP ${res.statusCode}`));
          }
        });
      });

      req.on('error', (err) => {
        reject(err);
      });

      req.setTimeout(20000, () => {
        req.destroy();
        reject(new Error('Request timeout'));
      });
    });

    const endTime = Date.now();
    const responseTime = endTime - startTime;

    // Verify response contains 30 products
    const productCount = Array.isArray(responseData) ? responseData.length : 0;
    console.log(`Products returned: ${productCount}`);

    if (productCount !== 30) {
      console.warn(`Warning: Expected 30 products, got ${productCount}`);
    }

    // Calculate if test passed
    const passed = responseTime < 500 && queryCount <= 2;

    // Output results
    console.log('\n=== Performance Test Results ===');
    console.log(`Response Time: ${responseTime}ms (requirement: < 500ms)`);
    console.log(`Query Count: ${queryCount} (requirement: <= 2)`);
    console.log(`Test Passed: ${passed ? 'YES ✓' : 'NO ✗'}`);
    console.log('================================\n');

    // Write results to file
    const results = {
      response_time_ms: responseTime,
      query_count: queryCount,
      passed: passed
    };

    fs.writeFileSync(
      path.join('/workspace', 'solution.json'),
      JSON.stringify(results, null, 2)
    );

    console.log('Results written to /workspace/solution.json');

    // Cleanup and exit
    if (server && server.close) {
      server.close();
    }
    
    process.exit(passed ? 0 : 1);

  } catch (error) {
    console.error('Test failed with error:', error.message);
    
    // Write error results
    const errorResults = {
      response_time_ms: 999999,
      query_count: 999,
      passed: false
    };
    
    fs.writeFileSync(
      path.join('/workspace', 'solution.json'),
      JSON.stringify(errorResults, null, 2)
    );

    if (server && server.close) {
      server.close();
    }
    
    process.exit(1);
  }
}

runTest();