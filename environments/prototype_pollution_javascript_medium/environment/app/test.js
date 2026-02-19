const express = require('express');
const assert = require('assert');
const preferences = require('./preferences');
const utils = require('./utils');

// Helper function to clean up Object.prototype between tests
function cleanPrototype() {
  delete Object.prototype.isAdmin;
  delete Object.prototype.polluted;
  delete Object.prototype.injected;
}

// Test counter
let passedTests = 0;
let failedTests = 0;

function test(name, fn) {
  try {
    cleanPrototype();
    fn();
    console.log(`✓ ${name}`);
    passedTests++;
  } catch (error) {
    console.log(`✗ ${name}`);
    console.log(`  Error: ${error.message}`);
    failedTests++;
  }
}

console.log('\n=== LEGITIMATE FUNCTIONALITY TESTS ===\n');

test('Update simple preferences', () => {
  preferences.clearAll(); // Reset for test
  const userId = 'user1';
  const prefs = { theme: 'dark', language: 'en' };
  
  preferences.updatePreferences(userId, prefs);
  const result = preferences.getPreferences(userId);
  
  assert.strictEqual(result.theme, 'dark');
  assert.strictEqual(result.language, 'en');
});

test('Update nested preferences', () => {
  preferences.clearAll();
  const userId = 'user2';
  const prefs = {
    settings: {
      notifications: true,
      email: 'test@example.com'
    }
  };
  
  preferences.updatePreferences(userId, prefs);
  const result = preferences.getPreferences(userId);
  
  assert.strictEqual(result.settings.notifications, true);
  assert.strictEqual(result.settings.email, 'test@example.com');
});

test('Multiple updates to same user merge correctly', () => {
  preferences.clearAll();
  const userId = 'user3';
  
  preferences.updatePreferences(userId, { theme: 'dark' });
  preferences.updatePreferences(userId, { language: 'en' });
  preferences.updatePreferences(userId, { theme: 'light' });
  
  const result = preferences.getPreferences(userId);
  
  assert.strictEqual(result.theme, 'light');
  assert.strictEqual(result.language, 'en');
});

test('Different users have separate preferences', () => {
  preferences.clearAll();
  
  preferences.updatePreferences('user4', { theme: 'dark' });
  preferences.updatePreferences('user5', { theme: 'light' });
  
  const result4 = preferences.getPreferences('user4');
  const result5 = preferences.getPreferences('user5');
  
  assert.strictEqual(result4.theme, 'dark');
  assert.strictEqual(result5.theme, 'light');
});

test('Deep nested object updates work correctly', () => {
  preferences.clearAll();
  const userId = 'user6';
  const prefs = {
    profile: {
      personal: {
        name: 'John',
        age: 30
      },
      settings: {
        privacy: 'public'
      }
    }
  };
  
  preferences.updatePreferences(userId, prefs);
  const result = preferences.getPreferences(userId);
  
  assert.strictEqual(result.profile.personal.name, 'John');
  assert.strictEqual(result.profile.settings.privacy, 'public');
});

test('Utils.merge merges objects correctly', () => {
  const target = { a: 1, b: 2 };
  const source = { b: 3, c: 4 };
  
  utils.merge(target, source);
  
  assert.strictEqual(target.a, 1);
  assert.strictEqual(target.b, 3);
  assert.strictEqual(target.c, 4);
});

console.log('\n=== PROTOTYPE POLLUTION VULNERABILITY TESTS ===\n');

test('SECURITY: __proto__ pollution attempt should be blocked', () => {
  preferences.clearAll();
  cleanPrototype();
  
  const userId = 'attacker1';
  const maliciousPrefs = JSON.parse('{"__proto__": {"isAdmin": true}}');
  
  preferences.updatePreferences(userId, maliciousPrefs);
  
  // Check that pollution did NOT occur
  const testObj = {};
  assert.strictEqual(testObj.isAdmin, undefined, 'Object.prototype was polluted via __proto__!');
});

test('SECURITY: constructor.prototype pollution attempt should be blocked', () => {
  preferences.clearAll();
  cleanPrototype();
  
  const userId = 'attacker2';
  const maliciousPrefs = {
    constructor: {
      prototype: {
        isAdmin: true
      }
    }
  };
  
  preferences.updatePreferences(userId, maliciousPrefs);
  
  // Check that pollution did NOT occur
  const testObj = {};
  assert.strictEqual(testObj.isAdmin, undefined, 'Object.prototype was polluted via constructor.prototype!');
});

test('SECURITY: nested __proto__ pollution should be blocked', () => {
  preferences.clearAll();
  cleanPrototype();
  
  const userId = 'attacker3';
  const maliciousPrefs = {
    settings: JSON.parse('{"__proto__": {"polluted": true}}')
  };
  
  preferences.updatePreferences(userId, maliciousPrefs);
  
  // Check that pollution did NOT occur
  const testObj = {};
  assert.strictEqual(testObj.polluted, undefined, 'Object.prototype was polluted via nested __proto__!');
});

test('SECURITY: Utils.merge should block __proto__ pollution', () => {
  cleanPrototype();
  
  const target = {};
  const maliciousSource = JSON.parse('{"__proto__": {"injected": true}}');
  
  utils.merge(target, maliciousSource);
  
  // Check that pollution did NOT occur
  const testObj = {};
  assert.strictEqual(testObj.injected, undefined, 'Object.prototype was polluted via utils.merge!');
});

test('SECURITY: prototype key should be blocked', () => {
  preferences.clearAll();
  cleanPrototype();
  
  const userId = 'attacker4';
  const maliciousPrefs = {
    prototype: {
      isAdmin: true
    }
  };
  
  preferences.updatePreferences(userId, maliciousPrefs);
  
  // Check that pollution did NOT occur on Object
  const testObj = {};
  assert.strictEqual(testObj.isAdmin, undefined, 'Object.prototype was polluted via prototype key!');
});

test('SECURITY: Deep nested constructor pollution should be blocked', () => {
  preferences.clearAll();
  cleanPrototype();
  
  const userId = 'attacker5';
  const maliciousPrefs = {
    level1: {
      level2: {
        constructor: {
          prototype: {
            isAdmin: true
          }
        }
      }
    }
  };
  
  preferences.updatePreferences(userId, maliciousPrefs);
  
  // Check that pollution did NOT occur
  const testObj = {};
  assert.strictEqual(testObj.isAdmin, undefined, 'Object.prototype was polluted via deep nested constructor!');
});

test('SECURITY: Array with __proto__ should be blocked', () => {
  preferences.clearAll();
  cleanPrototype();
  
  const userId = 'attacker6';
  const maliciousPrefs = {
    items: [
      { name: 'item1' },
      JSON.parse('{"__proto__": {"polluted": true}}')
    ]
  };
  
  preferences.updatePreferences(userId, maliciousPrefs);
  
  // Check that pollution did NOT occur
  const testObj = {};
  assert.strictEqual(testObj.polluted, undefined, 'Object.prototype was polluted via array element!');
});

console.log('\n=== TEST SUMMARY ===\n');
console.log(`Passed: ${passedTests}`);
console.log(`Failed: ${failedTests}`);
console.log(`Total: ${passedTests + failedTests}\n`);

if (failedTests === 0) {
  console.log('✓ All tests passed!');
  process.exit(0);
} else {
  console.log('✗ Some tests failed!');
  process.exit(1);
}