const WebSocket = require('ws');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

class ReconnectionTestHarness {
  constructor() {
    this.server = null;
    this.client = null;
    this.connectionAttempts = [];
    this.expectedDelays = [1000, 2000, 4000, 8000, 16000, 30000, 30000, 30000, 30000];
    this.tolerance = 200;
    this.maxAttempts = 10;
    this.currentAttempt = 0;
    this.testPassed = true;
    this.testStartTime = null;
  }

  log(message) {
    console.log(`[TEST HARNESS] ${message}`);
  }

  async startServer() {
    return new Promise((resolve) => {
      this.server = new WebSocket.Server({ port: 8080 });
      
      this.server.on('listening', () => {
        this.log('Mock WebSocket server started on ws://localhost:8080');
        resolve();
      });

      this.server.on('connection', (ws) => {
        const timestamp = Date.now();
        this.currentAttempt++;
        
        this.log(`Connection attempt #${this.currentAttempt} received at ${new Date(timestamp).toISOString()}`);
        this.connectionAttempts.push(timestamp);

        if (this.currentAttempt > 1) {
          const actualDelay = timestamp - this.connectionAttempts[this.connectionAttempts.length - 2];
          const expectedDelay = this.expectedDelays[this.currentAttempt - 2];
          const minDelay = expectedDelay - this.tolerance;
          const maxDelay = expectedDelay + this.tolerance;
          
          this.log(`  Measured delay: ${actualDelay}ms (expected: ${expectedDelay}ms ±${this.tolerance}ms)`);
          
          if (actualDelay < minDelay || actualDelay > maxDelay) {
            this.log(`  ❌ FAIL: Delay out of acceptable range [${minDelay}ms - ${maxDelay}ms]`);
            this.testPassed = false;
          } else {
            this.log(`  ✓ PASS: Delay within acceptable range`);
          }
        }

        // Simulate connection failure by immediately closing
        if (this.currentAttempt <= this.maxAttempts) {
          setTimeout(() => {
            this.log(`Closing connection #${this.currentAttempt} to simulate failure`);
            ws.close();
          }, 100);
        }
      });

      this.server.on('error', (error) => {
        this.log(`Server error: ${error.message}`);
      });
    });
  }

  async startClient() {
    return new Promise((resolve, reject) => {
      this.log('Starting client process...');
      
      this.client = spawn('node', ['/workspace/ws-client.js'], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      this.client.stdout.on('data', (data) => {
        console.log(`[CLIENT] ${data.toString().trim()}`);
      });

      this.client.stderr.on('data', (data) => {
        console.error(`[CLIENT ERROR] ${data.toString().trim()}`);
      });

      this.client.on('error', (error) => {
        this.log(`Failed to start client: ${error.message}`);
        reject(error);
      });

      this.client.on('exit', (code) => {
        this.log(`Client process exited with code ${code}`);
        resolve(code);
      });

      // Give client time to start
      setTimeout(() => resolve(), 500);
    });
  }

  async runTest() {
    this.log('=== WebSocket Reconnection Test Harness ===\n');
    
    try {
      // Start server first
      await this.startServer();
      
      // Start client
      this.testStartTime = Date.now();
      const clientPromise = this.startClient();
      
      // Wait for client to complete (should stop after 10 attempts)
      // Maximum test duration: initial + 9 delays + buffer
      // 0 + 1 + 2 + 4 + 8 + 16 + 30 + 30 + 30 + 30 = 151 seconds + buffer
      const maxTestDuration = 180000; // 180 seconds
      
      await Promise.race([
        clientPromise,
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Test timeout')), maxTestDuration)
        )
      ]);

      // Give a moment for final logging
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Analyze results
      this.log('\n=== Test Results ===');
      this.log(`Total connection attempts: ${this.currentAttempt}`);
      
      if (this.currentAttempt !== this.maxAttempts) {
        this.log(`❌ FAIL: Expected exactly ${this.maxAttempts} attempts, got ${this.currentAttempt}`);
        this.testPassed = false;
      } else {
        this.log(`✓ PASS: Client attempted exactly ${this.maxAttempts} connections`);
      }

      if (this.testPassed) {
        this.log('\n✓ ALL TESTS PASSED');
      } else {
        this.log('\n❌ SOME TESTS FAILED');
      }

      // Generate solution.json
      const solution = {
        passed: this.testPassed,
        max_attempts: this.currentAttempt
      };

      const solutionPath = path.join('/workspace', 'solution.json');
      fs.writeFileSync(solutionPath, JSON.stringify(solution, null, 2));
      this.log(`\nSolution file generated at ${solutionPath}`);
      this.log(JSON.stringify(solution, null, 2));

    } catch (error) {
      this.log(`Test error: ${error.message}`);
      
      // Generate failure solution
      const solution = {
        passed: false,
        max_attempts: this.currentAttempt
      };
      
      const solutionPath = path.join('/workspace', 'solution.json');
      fs.writeFileSync(solutionPath, JSON.stringify(solution, null, 2));
      
    } finally {
      this.cleanup();
    }
  }

  cleanup() {
    this.log('\nCleaning up...');
    
    if (this.client && !this.client.killed) {
      this.log('Terminating client process');
      this.client.kill();
    }

    if (this.server) {
      this.log('Closing server');
      this.server.close();
    }

    this.log('Test harness complete');
  }
}

// Run the test harness
const harness = new ReconnectionTestHarness();
harness.runTest().catch((error) => {
  console.error('Fatal test error:', error);
  process.exit(1);
});