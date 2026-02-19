const io = require('socket.io-client');
const jwt = require('jsonwebtoken');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const JWT_SECRET = 'test-secret-key';

// Helper function to generate JWT tokens
function generateToken(role) {
    return jwt.sign({ role: role, user: 'test-user' }, JWT_SECRET);
}

// Helper function to test a connection
function testConnection(namespace, token, timeout = 5000) {
    return new Promise((resolve) => {
        const options = {
            transports: ['websocket'],
            reconnection: false
        };
        
        if (token) {
            options.auth = { token: token };
        }
        
        const socket = io(`http://localhost:3000${namespace}`, options);
        
        const timer = setTimeout(() => {
            socket.close();
            resolve({ success: false, reason: 'timeout' });
        }, timeout);
        
        socket.on('connect', () => {
            clearTimeout(timer);
            socket.close();
            resolve({ success: true, socket: socket });
        });
        
        socket.on('connect_error', (error) => {
            clearTimeout(timer);
            socket.close();
            resolve({ success: false, reason: error.message });
        });
    });
}

async function runTests() {
    let serverProcess = null;
    
    try {
        console.log('Starting server...');
        
        // Start the server
        serverProcess = spawn('node', ['../server/server.js'], {
            cwd: __dirname,
            stdio: ['ignore', 'pipe', 'pipe']
        });
        
        serverProcess.stdout.on('data', (data) => {
            console.log(`Server: ${data}`);
        });
        
        serverProcess.stderr.on('data', (data) => {
            console.error(`Server Error: ${data}`);
        });
        
        // Wait for server to start
        await new Promise(resolve => setTimeout(resolve, 2000));
        console.log('Server started\n');
        
        // Initialize results
        const results = {
            public_accessible: false,
            admin_protected: false,
            analytics_protected: false,
            all_tests_passed: false
        };
        
        // Test 1: /public namespace without token (should succeed)
        console.log('Testing /public namespace without token...');
        const publicTest = await testConnection('/public', null);
        results.public_accessible = publicTest.success;
        console.log(`Result: ${publicTest.success ? 'PASS' : 'FAIL'} - ${publicTest.success ? 'Connected successfully' : publicTest.reason}\n`);
        
        // Test 2: /admin namespace with 'user' role (should fail)
        console.log('Testing /admin namespace with user role...');
        const adminUserTest = await testConnection('/admin', generateToken('user'));
        const adminUserRejected = !adminUserTest.success;
        console.log(`Result: ${adminUserRejected ? 'PASS' : 'FAIL'} - ${adminUserRejected ? 'Correctly rejected' : 'Incorrectly accepted'}\n`);
        
        // Test 3: /admin namespace with 'admin' role (should succeed)
        console.log('Testing /admin namespace with admin role...');
        const adminAdminTest = await testConnection('/admin', generateToken('admin'));
        const adminAdminAccepted = adminAdminTest.success;
        console.log(`Result: ${adminAdminAccepted ? 'PASS' : 'FAIL'} - ${adminAdminAccepted ? 'Connected successfully' : adminAdminTest.reason}\n`);
        
        results.admin_protected = adminUserRejected && adminAdminAccepted;
        
        // Test 4: /analytics namespace with 'user' role (should fail)
        console.log('Testing /analytics namespace with user role...');
        const analyticsUserTest = await testConnection('/analytics', generateToken('user'));
        const analyticsUserRejected = !analyticsUserTest.success;
        console.log(`Result: ${analyticsUserRejected ? 'PASS' : 'FAIL'} - ${analyticsUserRejected ? 'Correctly rejected' : 'Incorrectly accepted'}\n`);
        
        // Test 5: /analytics namespace with 'analyst' role (should succeed)
        console.log('Testing /analytics namespace with analyst role...');
        const analyticsAnalystTest = await testConnection('/analytics', generateToken('analyst'));
        const analyticsAnalystAccepted = analyticsAnalystTest.success;
        console.log(`Result: ${analyticsAnalystAccepted ? 'PASS' : 'FAIL'} - ${analyticsAnalystAccepted ? 'Connected successfully' : analyticsAnalystTest.reason}\n`);
        
        // Test 6: /analytics namespace with 'admin' role (should succeed)
        console.log('Testing /analytics namespace with admin role...');
        const analyticsAdminTest = await testConnection('/analytics', generateToken('admin'));
        const analyticsAdminAccepted = analyticsAdminTest.success;
        console.log(`Result: ${analyticsAdminAccepted ? 'PASS' : 'FAIL'} - ${analyticsAdminAccepted ? 'Connected successfully' : analyticsAdminTest.reason}\n`);
        
        results.analytics_protected = analyticsUserRejected && analyticsAnalystAccepted && analyticsAdminAccepted;
        
        // Determine overall result
        results.all_tests_passed = results.public_accessible && results.admin_protected && results.analytics_protected;
        
        // Write results to file
        const resultsPath = path.join('/workspace', 'auth-results.json');
        fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
        
        console.log('=================================');
        console.log('Test Results:');
        console.log(`Public Accessible: ${results.public_accessible ? 'PASS' : 'FAIL'}`);
        console.log(`Admin Protected: ${results.admin_protected ? 'PASS' : 'FAIL'}`);
        console.log(`Analytics Protected: ${results.analytics_protected ? 'PASS' : 'FAIL'}`);
        console.log('=================================');
        
        if (results.all_tests_passed) {
            console.log('✓ All tests passed!');
        } else {
            console.log('✗ Some tests failed');
        }
        
        // Cleanup
        if (serverProcess) {
            serverProcess.kill();
        }
        
        process.exit(results.all_tests_passed ? 0 : 1);
        
    } catch (error) {
        console.error('Error during testing:', error);
        
        // Cleanup on error
        if (serverProcess) {
            serverProcess.kill();
        }
        
        process.exit(1);
    }
}

runTests();