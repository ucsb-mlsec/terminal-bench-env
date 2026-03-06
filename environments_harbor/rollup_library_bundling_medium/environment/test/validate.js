const path = require('path');
const vm = require('vm');
const fs = require('fs');

// Test data
const tests = {
  capitalize: { input: ['hello'], expected: 'Hello' },
  reverse: { input: ['abc'], expected: 'cba' },
  truncate: { input: ['hello world', 8], expected: 'hello...' },
  toCamelCase: { input: ['hello-world'], expected: 'helloWorld' },
  chunk: { input: [[1, 2, 3, 4, 5], 2], expected: [[1, 2], [3, 4], [5]] },
  flatten: { input: [[1, [2, 3], 4]], expected: [1, 2, 3, 4] },
  unique: { input: [[1, 2, 2, 3, 3, 3]], expected: [1, 2, 3] },
  sum: { input: [[1, 2, 3, 4]], expected: 10 }
};

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;

function deepEqual(a, b) {
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!deepEqual(a[i], b[i])) return false;
    }
    return true;
  }
  return false;
}

function runTests(lib, formatName) {
  console.log(`\n=== Testing ${formatName} ===`);
  let formatPassed = 0;
  let formatFailed = 0;

  for (const [funcName, testData] of Object.entries(tests)) {
    totalTests++;
    try {
      const result = lib[funcName](...testData.input);
      const passed = deepEqual(result, testData.expected);
      
      if (passed) {
        console.log(`✓ ${funcName}: PASS`);
        passedTests++;
        formatPassed++;
      } else {
        console.log(`✗ ${funcName}: FAIL - Expected ${JSON.stringify(testData.expected)}, got ${JSON.stringify(result)}`);
        failedTests++;
        formatFailed++;
      }
    } catch (error) {
      console.log(`✗ ${funcName}: ERROR - ${error.message}`);
      failedTests++;
      formatFailed++;
    }
  }

  console.log(`${formatName} Results: ${formatPassed}/${Object.keys(tests).length} passed`);
  return formatFailed === 0;
}

async function testCommonJS() {
  console.log('\n--- Testing CommonJS Bundle ---');
  try {
    const cjsPath = path.join(__dirname, '../dist/util-lib.cjs.js');
    if (!fs.existsSync(cjsPath)) {
      console.log('✗ CommonJS bundle not found');
      return false;
    }
    delete require.cache[require.resolve(cjsPath)];
    const lib = require(cjsPath);
    return runTests(lib, 'CommonJS');
  } catch (error) {
    console.log(`✗ CommonJS Error: ${error.message}`);
    return false;
  }
}

async function testESM() {
  console.log('\n--- Testing ESM Bundle ---');
  try {
    const esmPath = path.join(__dirname, '../dist/util-lib.esm.js');
    if (!fs.existsSync(esmPath)) {
      console.log('✗ ESM bundle not found');
      return false;
    }
    const lib = await import('file://' + esmPath);
    return runTests(lib, 'ESM');
  } catch (error) {
    console.log(`✗ ESM Error: ${error.message}`);
    return false;
  }
}

async function testUMD() {
  console.log('\n--- Testing UMD Bundle ---');
  try {
    const umdPath = path.join(__dirname, '../dist/util-lib.umd.js');
    if (!fs.existsSync(umdPath)) {
      console.log('✗ UMD bundle not found');
      return false;
    }

    const code = fs.readFileSync(umdPath, 'utf-8');
    const sandbox = { window: {}, module: {}, exports: {} };
    vm.createContext(sandbox);
    vm.runInContext(code, sandbox);

    // UMD can attach to window, exports, or module.exports
    const lib = sandbox.window.UtilLib || sandbox.module.exports || sandbox.exports;
    
    if (!lib) {
      console.log('✗ UMD library not found in global scope');
      return false;
    }

    return runTests(lib, 'UMD');
  } catch (error) {
    console.log(`✗ UMD Error: ${error.message}`);
    return false;
  }
}

async function main() {
  console.log('Starting validation tests...\n');

  const cjsPass = await testCommonJS();
  const esmPass = await testESM();
  const umdPass = await testUMD();

  console.log('\n' + '='.repeat(50));
  console.log(`Total Tests: ${totalTests}`);
  console.log(`Passed: ${passedTests}`);
  console.log(`Failed: ${failedTests}`);
  console.log('='.repeat(50));

  if (cjsPass && esmPass && umdPass) {
    console.log('\n✓ All bundles passed validation!');
    process.exit(0);
  } else {
    console.log('\n✗ Some bundles failed validation');
    process.exit(1);
  }
}

main().catch(error => {
  console.error('Validation script error:', error);
  process.exit(1);
});