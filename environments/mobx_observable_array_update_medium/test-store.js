const TodoStore = require('./todo-store.js');

console.log('Starting TodoStore test...\n');

// Create a new store instance
const store = new TodoStore();

// Log initial count
const initialCount = store.todoCount;
console.log(`Initial count: ${initialCount}`);

// Add first todo
store.addTodo('Buy groceries');
console.log('Added todo: Buy groceries');
console.log(`Current count: ${store.todoCount}`);

// Add second todo
store.addTodo('Write documentation');
console.log('Added todo: Write documentation');
console.log(`Current count: ${store.todoCount}`);

// Add third todo
store.addTodo('Review pull requests');
console.log('Added todo: Review pull requests');
console.log(`Current count: ${store.todoCount}`);

// Check final count
const expectedCount = initialCount + 3;
const actualCount = store.todoCount;

console.log(`\nExpected count: ${expectedCount}`);
console.log(`Actual count: ${actualCount}`);

if (actualCount === expectedCount) {
    console.log('\n✓ TEST PASSED');
    process.exit(0);
} else {
    console.log(`\n✗ TEST FAILED: expected ${expectedCount} but got ${actualCount}`);
    process.exit(1);
}