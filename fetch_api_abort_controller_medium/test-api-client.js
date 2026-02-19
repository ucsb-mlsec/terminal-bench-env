const http = require('http');
const assert = require('assert');

// Test configuration
let testsPassed = 0;
let testsFailed = 0;
const TEST_PORT = 8765;

// Mock server for testing
let mockServer = null;

function createMockServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url, `http://localhost:${TEST_PORT}`);
      const delay = parseInt(url.searchParams.get('delay') || '0');
      const hang = url.searchParams.get('hang') === 'true';

      if (hang) {
        // Don't respond at all - simulate hanging endpoint
        return;
      }

      setTimeout(() => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, delay }));
      }, delay);
    });

    server.listen(TEST_PORT, () => {
      resolve(server);
    });
  });
}

function logTest(testName, passed, error = null) {
  if (passed) {
    console.log(`✓ PASS: ${testName}`);
    testsPassed++;
  } else {
    console.log(`✗ FAIL: ${testName}`);
    if (error) {
      console.log(`  Error: ${error.message}`);
    }
    testsFailed++;
  }
}

async function test1_FastRequestSucceeds() {
  const testName = 'Test 1: Request that completes quickly (should succeed)';
  try {
    const ApiClient = require('./api-client-fixed.js');
    const client = new ApiClient();
    
    const result = await client.request(
      `http://localhost:${TEST_PORT}?delay=50`,
      { timeout: 1000 }
    );
    
    assert.strictEqual(result.success, true);
    assert.strictEqual(result.delay, 50);
    logTest(testName, true);
  } catch (error) {
    logTest(testName, false, error);
  }
}

async function test2_RequestExceedsTimeout() {
  const testName = 'Test 2: Request that exceeds timeout (should abort and throw timeout error)';
  try {
    const ApiClient = require('./api-client-fixed.js');
    const client = new ApiClient();
    
    let timeoutErrorCaught = false;
    let errorMessage = '';
    
    try {
      await client.request(
        `http://localhost:${TEST_PORT}?delay=500`,
        { timeout: 100 }
      );
    } catch (error) {
      timeoutErrorCaught = true;
      errorMessage = error.message;
    }
    
    assert.strictEqual(timeoutErrorCaught, true, 'Should throw timeout error');
    assert.ok(
      errorMessage.toLowerCase().includes('timeout') || 
      errorMessage.toLowerCase().includes('abort'),
      'Error message should mention timeout or abort'
    );
    
    logTest(testName, true);
  } catch (error) {
    logTest(testName, false, error);
  }
}

async function test3_MultipleConcurrentRequests() {
  const testName = 'Test 3: Multiple concurrent requests with different timeouts (verify independent handling)';
  try {
    const ApiClient = require('./api-client-fixed.js');
    const client = new ApiClient();
    
    const results = await Promise.allSettled([
      client.request(`http://localhost:${TEST_PORT}?delay=50`, { timeout: 200 }),
      client.request(`http://localhost:${TEST_PORT}?delay=300`, { timeout: 100 }),
      client.request(`http://localhost:${TEST_PORT}?delay=80`, { timeout: 500 })
    ]);
    
    // First request should succeed (50ms delay, 200ms timeout)
    assert.strictEqual(results[0].status, 'fulfilled', 'First request should succeed');
    assert.strictEqual(results[0].value.delay, 50);
    
    // Second request should timeout (300ms delay, 100ms timeout)
    assert.strictEqual(results[1].status, 'rejected', 'Second request should timeout');
    
    // Third request should succeed (80ms delay, 500ms timeout)
    assert.strictEqual(results[2].status, 'fulfilled', 'Third request should succeed');
    assert.strictEqual(results[2].value.delay, 80);
    
    logTest(testName, true);
  } catch (error) {
    logTest(testName, false, error);
  }
}

async function test4_ProperResourceCleanup() {
  const testName = 'Test 4: Request that times out should properly clean up resources';
  try {
    const ApiClient = require('./api-client-fixed.js');
    const client = new ApiClient();
    
    // Make a request that will timeout
    try {
      await client.request(
        `http://localhost:${TEST_PORT}?hang=true`,
        { timeout: 100 }
      );
    } catch (error) {
      // Expected to fail
    }
    
    // Wait a bit to ensure cleanup happens
    await new Promise(resolve => setTimeout(resolve, 50));
    
    // Make another request to verify the client still works
    const result = await client.request(
      `http://localhost:${TEST_PORT}?delay=50`,
      { timeout: 500 }
    );
    
    assert.strictEqual(result.success, true);
    logTest(testName, true);
  } catch (error) {
    logTest(testName, false, error);
  }
}

async function test5_MeaningfulErrorMessage() {
  const testName = 'Test 5: Timeout error should contain meaningful error message';
  try {
    const ApiClient = require('./api-client-fixed.js');
    const client = new ApiClient();
    
    let error = null;
    try {
      await client.request(
        `http://localhost:${TEST_PORT}?hang=true`,
        { timeout: 150 }
      );
    } catch (e) {
      error = e;
    }
    
    assert.ok(error !== null, 'Should throw an error');
    assert.ok(error.message, 'Error should have a message');
    assert.ok(
      error.message.length > 10,
      'Error message should be descriptive'
    );
    assert.ok(
      error.message.toLowerCase().includes('timeout') ||
      error.message.toLowerCase().includes('abort') ||
      error.message.toLowerCase().includes('exceed'),
      'Error message should indicate timeout/abort'
    );
    
    logTest(testName, true);
  } catch (error) {
    logTest(testName, false, error);
  }
}

async function runAllTests() {
  console.log('Starting API Client Timeout Tests...\n');
  
  try {
    // Start mock server
    mockServer = await createMockServer();
    console.log(`Mock server started on port ${TEST_PORT}\n`);
    
    // Run all tests
    await test1_FastRequestSucceeds();
    await test2_RequestExceedsTimeout();
    await test3_MultipleConcurrentRequests();
    await test4_ProperResourceCleanup();
    await test5_MeaningfulErrorMessage();
    
  } catch (error) {
    console.error('Test suite error:', error);
    testsFailed++;
  } finally {
    // Cleanup
    if (mockServer) {
      mockServer.close();
    }
  }
  
  // Print results
  console.log('\n' + '='.repeat(50));
  console.log('Test Results:');
  console.log('='.repeat(50));
  console.log(`TESTS_PASSED=${testsPassed}`);
  console.log(`TESTS_FAILED=${testsFailed}`);
  console.log(`STATUS=${testsFailed === 0 ? 'PASS' : 'FAIL'}`);
  
  process.exit(testsFailed === 0 ? 0 : 1);
}

// Run tests
runAllTests().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});