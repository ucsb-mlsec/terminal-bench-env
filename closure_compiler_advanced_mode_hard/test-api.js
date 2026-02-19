const fs = require('fs');
const vm = require('vm');

// Check if the compiled file exists
if (!fs.existsSync('/workspace/calculator.min.js')) {
    console.error('ERROR: Compiled file not found at /workspace/calculator.min.js');
    process.exit(1);
}

// Read and execute the compiled JavaScript
const code = fs.readFileSync('/workspace/calculator.min.js', 'utf8');
const sandbox = { console, module, exports, require };
vm.createContext(sandbox);
vm.runInContext(code, sandbox);

// Access the Calculator object from the sandbox
const Calculator = sandbox.Calculator || global.Calculator;

if (!Calculator) {
    console.error('ERROR: Calculator object not found in compiled output');
    process.exit(1);
}

// Test the public API methods
let allTestsPassed = true;

// Test add method
if (typeof Calculator.add !== 'function') {
    console.error('FAIL: add method is not accessible');
    allTestsPassed = false;
} else {
    const result = Calculator.add(5, 3);
    if (result === 8) {
        console.log('PASS: add(5, 3) = 8');
    } else {
        console.error(`FAIL: add(5, 3) expected 8, got ${result}`);
        allTestsPassed = false;
    }
}

// Test subtract method
if (typeof Calculator.subtract !== 'function') {
    console.error('FAIL: subtract method is not accessible');
    allTestsPassed = false;
} else {
    const result = Calculator.subtract(10, 4);
    if (result === 6) {
        console.log('PASS: subtract(10, 4) = 6');
    } else {
        console.error(`FAIL: subtract(10, 4) expected 6, got ${result}`);
        allTestsPassed = false;
    }
}

// Test multiply method
if (typeof Calculator.multiply !== 'function') {
    console.error('FAIL: multiply method is not accessible');
    allTestsPassed = false;
} else {
    const result = Calculator.multiply(6, 7);
    if (result === 42) {
        console.log('PASS: multiply(6, 7) = 42');
    } else {
        console.error(`FAIL: multiply(6, 7) expected 42, got ${result}`);
        allTestsPassed = false;
    }
}

// Test divide method
if (typeof Calculator.divide !== 'function') {
    console.error('FAIL: divide method is not accessible');
    allTestsPassed = false;
} else {
    const result = Calculator.divide(20, 4);
    if (result === 5) {
        console.log('PASS: divide(20, 4) = 5');
    } else {
        console.error(`FAIL: divide(20, 4) expected 5, got ${result}`);
        allTestsPassed = false;
    }
}

// Exit with appropriate code
if (allTestsPassed) {
    console.log('\nAll tests passed!');
    process.exit(0);
} else {
    console.error('\nSome tests failed!');
    process.exit(1);
}