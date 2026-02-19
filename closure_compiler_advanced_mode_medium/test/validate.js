// test/validate.js
// Validation tests for the compiled utility library

const fs = require('fs');
const path = require('path');

// Load the compiled library
let utils;
try {
    const compiledPath = path.join(__dirname, '../dist/utils.min.js');
    if (!fs.existsSync(compiledPath)) {
        throw new Error('Compiled file not found at ' + compiledPath);
    }
    
    const code = fs.readFileSync(compiledPath, 'utf8');
    // Execute the compiled code in a way that captures exports
    const module = { exports: {} };
    const exports = module.exports;
    eval(code);
    utils = module.exports.default || module.exports || global.utils || window?.utils;
    
    if (!utils) {
        // Try alternative loading method
        delete require.cache[require.resolve('../dist/utils.min.js')];
        utils = require('../dist/utils.min.js');
    }
} catch (error) {
    console.error('Failed to load compiled library:', error.message);
    process.exit(1);
}

let testsPassed = 0;
let testsFailed = 0;

function assert(condition, message) {
    if (!condition) {
        throw new Error('Assertion failed: ' + message);
    }
}

function assertEquals(actual, expected, message) {
    const actualStr = JSON.stringify(actual);
    const expectedStr = JSON.stringify(expected);
    if (actualStr !== expectedStr) {
        throw new Error(`${message}\n  Expected: ${expectedStr}\n  Actual: ${actualStr}`);
    }
}

function test(name, fn) {
    try {
        fn();
        console.log('✓ ' + name);
        testsPassed++;
    } catch (error) {
        console.error('✗ ' + name);
        console.error('  ' + error.message);
        testsFailed++;
    }
}

console.log('Starting validation tests...\n');

// String Utilities Tests
test('capitalize: converts first letter to uppercase', () => {
    assertEquals(utils.capitalize('hello'), 'Hello', 'capitalize("hello") should return "Hello"');
    assertEquals(utils.capitalize('world'), 'World', 'capitalize("world") should return "World"');
    assertEquals(utils.capitalize(''), '', 'capitalize("") should return ""');
});

test('truncate: shortens strings correctly', () => {
    const longString = 'This is a very long string that needs to be truncated';
    const result = utils.truncate(longString, 20);
    assert(result.length <= 23, 'truncated string should be at most 23 chars (20 + ...)');
    assert(result.includes('...'), 'truncated string should include ...');
});

test('truncate: does not modify short strings', () => {
    assertEquals(utils.truncate('short', 10), 'short', 'short string should not be truncated');
});

// Array Utilities Tests
test('unique: removes duplicate values', () => {
    assertEquals(utils.unique([1, 2, 2, 3, 3, 3]), [1, 2, 3], 'should remove duplicates from number array');
    assertEquals(utils.unique(['a', 'b', 'a', 'c']), ['a', 'b', 'c'], 'should remove duplicates from string array');
});

test('flatten: works on nested arrays', () => {
    assertEquals(utils.flatten([1, [2, 3], [4, [5]]]), [1, 2, 3, 4, 5], 'should flatten nested arrays');
    assertEquals(utils.flatten([1, 2, 3]), [1, 2, 3], 'should handle flat arrays');
});

test('chunk: splits arrays into chunks', () => {
    const result = utils.chunk([1, 2, 3, 4, 5], 2);
    assertEquals(result, [[1, 2], [3, 4], [5]], 'should split array into chunks of 2');
});

// Object Utilities Tests
test('deepClone: creates independent copies', () => {
    const original = { a: 1, b: { c: 2 } };
    const cloned = utils.deepClone(original);
    cloned.b.c = 999;
    assertEquals(original.b.c, 2, 'original should not be modified');
    assertEquals(cloned.b.c, 999, 'clone should be modified');
});

test('merge: combines objects correctly', () => {
    const obj1 = { a: 1, b: 2 };
    const obj2 = { b: 3, c: 4 };
    const result = utils.merge(obj1, obj2);
    assertEquals(result, { a: 1, b: 3, c: 4 }, 'should merge objects with override');
});

test('pick: extracts specified properties', () => {
    const obj = { a: 1, b: 2, c: 3, d: 4 };
    const result = utils.pick(obj, ['a', 'c']);
    assertEquals(result, { a: 1, c: 3 }, 'should pick only specified properties');
});

// Validation Functions Tests
test('isEmail: validates email addresses correctly', () => {
    assert(utils.isEmail('test@example.com'), 'test@example.com should be valid');
    assert(utils.isEmail('user.name@domain.co.uk'), 'user.name@domain.co.uk should be valid');
    assert(!utils.isEmail('invalid.email'), 'invalid.email should be invalid');
    assert(!utils.isEmail('@example.com'), '@example.com should be invalid');
    assert(!utils.isEmail('test@'), 'test@ should be invalid');
});

test('isURL: validates URLs correctly', () => {
    assert(utils.isURL('https://example.com'), 'https://example.com should be valid');
    assert(utils.isURL('http://test.org/path'), 'http://test.org/path should be valid');
    assert(!utils.isURL('not a url'), 'not a url should be invalid');
    assert(!utils.isURL('ftp://example.com'), 'ftp://example.com should be invalid (if only http/https allowed)');
});

// Number Utilities Tests
test('clamp: constrains numbers to range', () => {
    assertEquals(utils.clamp(5, 0, 10), 5, 'value in range should stay same');
    assertEquals(utils.clamp(-5, 0, 10), 0, 'value below min should return min');
    assertEquals(utils.clamp(15, 0, 10), 10, 'value above max should return max');
});

test('random: generates random numbers in range', () => {
    for (let i = 0; i < 10; i++) {
        const result = utils.random(1, 10);
        assert(result >= 1 && result <= 10, 'random should be between 1 and 10');
    }
});

// Debounce/Throttle Tests
test('debounce: creates debounced function', () => {
    let counter = 0;
    const debounced = utils.debounce(() => { counter++; }, 50);
    assert(typeof debounced === 'function', 'debounce should return a function');
    debounced();
    debounced();
    debounced();
    assert(counter === 0, 'debounced function should not execute immediately');
});

console.log('\n' + '='.repeat(50));
console.log(`Tests passed: ${testsPassed}`);
console.log(`Tests failed: ${testsFailed}`);
console.log('='.repeat(50));

if (testsFailed > 0) {
    console.error('\nSome tests failed!');
    process.exit(1);
} else {
    console.log('\nALL TESTS PASSED');
    process.exit(0);
}