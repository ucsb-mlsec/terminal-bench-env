const merge = require('./merge-utils.js');

console.log('Running functionality tests for merge-utils.js...\n');

let testsPassed = 0;
let testsFailed = 0;

function assertEqual(actual, expected, testName) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    console.log(`FAILED: ${testName}`);
    console.log(`  Expected: ${JSON.stringify(expected)}`);
    console.log(`  Got: ${JSON.stringify(actual)}`);
    testsFailed++;
    return false;
  }
  console.log(`PASSED: ${testName}`);
  testsPassed++;
  return true;
}

// Test 1: Basic object merging
const obj1 = { a: 1, b: 2 };
const obj2 = { c: 3, d: 4 };
const result1 = merge(obj1, obj2);
assertEqual(result1, { a: 1, b: 2, c: 3, d: 4 }, 'Test 1: Basic object merging');

// Test 2: Deep nested object merging
const obj3 = { 
  user: { 
    name: 'John', 
    address: { city: 'NYC', zip: '10001' } 
  },
  status: 'active'
};
const obj4 = { 
  user: { 
    age: 30, 
    address: { country: 'USA' } 
  },
  role: 'admin'
};
const result2 = merge(obj3, obj4);
const expected2 = {
  user: {
    name: 'John',
    age: 30,
    address: { city: 'NYC', zip: '10001', country: 'USA' }
  },
  status: 'active',
  role: 'admin'
};
assertEqual(result2, expected2, 'Test 2: Deep nested object merging');

// Test 3: Property overwriting
const obj5 = { a: 1, b: 2, c: 3 };
const obj6 = { b: 99, d: 4 };
const result3 = merge(obj5, obj6);
assertEqual(result3, { a: 1, b: 99, c: 3, d: 4 }, 'Test 3: Property overwriting');

// Test 4: Merging with arrays
const obj7 = { items: [1, 2, 3], name: 'list1' };
const obj8 = { items: [4, 5], type: 'array' };
const result4 = merge(obj7, obj8);
assertEqual(result4, { items: [4, 5], name: 'list1', type: 'array' }, 'Test 4: Merging with arrays');

// Test 5: Merging empty objects
const obj9 = { a: 1, b: 2 };
const obj10 = {};
const result5 = merge(obj9, obj10);
assertEqual(result5, { a: 1, b: 2 }, 'Test 5: Merging with empty object');

// Test 6: Three-level deep nesting
const obj11 = {
  level1: {
    level2: {
      level3: {
        value: 'deep'
      }
    }
  }
};
const obj12 = {
  level1: {
    level2: {
      level3: {
        newValue: 'deeper'
      },
      extra: 'data'
    }
  }
};
const result6 = merge(obj11, obj12);
const expected6 = {
  level1: {
    level2: {
      level3: {
        value: 'deep',
        newValue: 'deeper'
      },
      extra: 'data'
    }
  }
};
assertEqual(result6, expected6, 'Test 6: Three-level deep nesting');

// Test 7: Multiple properties at different levels
const obj13 = { x: 1, nested: { y: 2 } };
const obj14 = { z: 3, nested: { w: 4 } };
const result7 = merge(obj13, obj14);
assertEqual(result7, { x: 1, z: 3, nested: { y: 2, w: 4 } }, 'Test 7: Multiple properties at different levels');

console.log(`\n========================================`);
console.log(`Tests Passed: ${testsPassed}`);
console.log(`Tests Failed: ${testsFailed}`);
console.log(`========================================\n`);

if (testsFailed > 0) {
  console.log('Some tests FAILED!');
  process.exit(1);
} else {
  console.log('All functionality tests PASSED');
  process.exit(0);
}