const mongoose = require('mongoose');
const User = require('../models/User');

let testsPassed = 0;
let testsFailed = 0;

// Connect to MongoDB
async function connectDB() {
    try {
        await mongoose.connect('mongodb://localhost:27017/testdb', {
            useNewUrlParser: true,
            useUnifiedTopology: true
        });
        console.log('Connected to MongoDB for testing\n');
    } catch (error) {
        console.error('MongoDB connection error:', error);
        process.exit(1);
    }
}

// Helper function to run tests
async function runTest(testName, testFunction) {
    try {
        await testFunction();
        console.log(`✓ PASS: ${testName}`);
        testsPassed++;
    } catch (error) {
        console.log(`✗ FAIL: ${testName}`);
        console.log(`  Error: ${error.message}\n`);
        testsFailed++;
    }
}

// Helper to expect validation error
async function expectValidationError(testName, userData) {
    try {
        const user = new User(userData);
        await user.save();
        throw new Error('Expected validation error but document was saved');
    } catch (error) {
        if (error.name === 'ValidationError' || error.message.includes('duplicate') || error.code === 11000) {
            return;
        }
        throw error;
    }
}

// Helper to expect successful save
async function expectSuccess(testName, userData) {
    const user = new User(userData);
    await user.save();
    await User.deleteOne({ _id: user._id });
}

// Test Suite
async function runTests() {
    console.log('=== Starting Mongoose Schema Validation Tests ===\n');

    // Clear test database
    await User.deleteMany({});

    // Test 1: Valid user creation
    await runTest('Valid user creation with all required fields', async () => {
        const validUser = {
            username: 'john_doe123',
            email: 'john@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        };
        await expectSuccess('valid user', validUser);
    });

    // Test 2: Valid user with optional fields
    await runTest('Valid user with optional fields', async () => {
        const validUser = {
            username: 'jane_smith',
            email: 'jane@example.com',
            age: 30,
            phoneNumber: '+12345678901',
            website: 'https://example.com',
            bio: 'This is a valid bio',
            role: 'moderator',
            accountStatus: 'active'
        };
        await expectSuccess('valid user with optional fields', validUser);
    });

    // Username validation tests
    await runTest('Username too short (2 characters)', async () => {
        await expectValidationError('short username', {
            username: 'ab',
            email: 'test1@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Username too long (31 characters)', async () => {
        await expectValidationError('long username', {
            username: 'a'.repeat(31),
            email: 'test2@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Username with spaces', async () => {
        await expectValidationError('username with spaces', {
            username: 'john doe',
            email: 'test3@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Username with special characters', async () => {
        await expectValidationError('username with special chars', {
            username: 'john@doe!',
            email: 'test4@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Username is required', async () => {
        await expectValidationError('missing username', {
            email: 'test5@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
    });

    // Email validation tests
    await runTest('Email without @ symbol', async () => {
        await expectValidationError('invalid email format', {
            username: 'testuser1',
            email: 'invalidemail.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Email without domain', async () => {
        await expectValidationError('email without domain', {
            username: 'testuser2',
            email: 'test@',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Email is required', async () => {
        await expectValidationError('missing email', {
            username: 'testuser3',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Email converted to lowercase', async () => {
        const user = new User({
            username: 'testuser4',
            email: 'TEST@EXAMPLE.COM',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
        await user.save();
        if (user.email !== 'test@example.com') {
            throw new Error('Email was not converted to lowercase');
        }
        await User.deleteOne({ _id: user._id });
    });

    // Age validation tests
    await runTest('Age too young (12)', async () => {
        await expectValidationError('age too young', {
            username: 'testuser5',
            email: 'test6@example.com',
            age: 12,
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Age too old (121)', async () => {
        await expectValidationError('age too old', {
            username: 'testuser6',
            email: 'test7@example.com',
            age: 121,
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Negative age', async () => {
        await expectValidationError('negative age', {
            username: 'testuser7',
            email: 'test8@example.com',
            age: -5,
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Age is required', async () => {
        await expectValidationError('missing age', {
            username: 'testuser8',
            email: 'test9@example.com',
            role: 'user',
            accountStatus: 'active'
        });
    });

    // Phone number validation tests
    await runTest('Invalid phone number format', async () => {
        await expectValidationError('invalid phone format', {
            username: 'testuser9',
            email: 'test10@example.com',
            age: 25,
            phoneNumber: '1234567890',
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Valid phone number E.164 format', async () => {
        await expectSuccess('valid phone', {
            username: 'testuser10',
            email: 'test11@example.com',
            age: 25,
            phoneNumber: '+12345678901',
            role: 'user',
            accountStatus: 'active'
        });
    });

    // Website validation tests
    await runTest('Invalid URL without protocol', async () => {
        await expectValidationError('invalid URL', {
            username: 'testuser11',
            email: 'test12@example.com',
            age: 25,
            website: 'example.com',
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Valid URL with https', async () => {
        await expectSuccess('valid https URL', {
            username: 'testuser12',
            email: 'test13@example.com',
            age: 25,
            website: 'https://example.com',
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Valid URL with http', async () => {
        await expectSuccess('valid http URL', {
            username: 'testuser13',
            email: 'test14@example.com',
            age: 25,
            website: 'http://example.com',
            role: 'user',
            accountStatus: 'active'
        });
    });

    // Bio validation tests
    await runTest('Bio exceeding 500 characters', async () => {
        await expectValidationError('bio too long', {
            username: 'testuser14',
            email: 'test15@example.com',
            age: 25,
            bio: 'a'.repeat(501),
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Valid bio with 500 characters', async () => {
        await expectSuccess('valid bio', {
            username: 'testuser15',
            email: 'test16@example.com',
            age: 25,
            bio: 'a'.repeat(500),
            role: 'user',
            accountStatus: 'active'
        });
    });

    // Role validation tests
    await runTest('Invalid role value', async () => {
        await expectValidationError('invalid role', {
            username: 'testuser16',
            email: 'test17@example.com',
            age: 25,
            role: 'superuser',
            accountStatus: 'active'
        });
    });

    await runTest('Valid role: user', async () => {
        await expectSuccess('role user', {
            username: 'testuser17',
            email: 'test18@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Valid role: moderator', async () => {
        await expectSuccess('role moderator', {
            username: 'testuser18',
            email: 'test19@example.com',
            age: 25,
            role: 'moderator',
            accountStatus: 'active'
        });
    });

    await runTest('Valid role: admin', async () => {
        await expectSuccess('role admin', {
            username: 'testuser19',
            email: 'test20@example.com',
            age: 25,
            role: 'admin',
            accountStatus: 'active'
        });
    });

    await runTest('Role is required', async () => {
        await expectValidationError('missing role', {
            username: 'testuser20',
            email: 'test21@example.com',
            age: 25,
            accountStatus: 'active'
        });
    });

    // Account status validation tests
    await runTest('Invalid accountStatus value', async () => {
        await expectValidationError('invalid account status', {
            username: 'testuser21',
            email: 'test22@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'pending'
        });
    });

    await runTest('Valid accountStatus: active', async () => {
        await expectSuccess('status active', {
            username: 'testuser22',
            email: 'test23@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
    });

    await runTest('Valid accountStatus: suspended', async () => {
        await expectSuccess('status suspended', {
            username: 'testuser23',
            email: 'test24@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'suspended'
        });
    });

    await runTest('Valid accountStatus: deleted', async () => {
        await expectSuccess('status deleted', {
            username: 'testuser24',
            email: 'test25@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'deleted'
        });
    });

    await runTest('AccountStatus is required', async () => {
        await expectValidationError('missing account status', {
            username: 'testuser25',
            email: 'test26@example.com',
            age: 25,
            role: 'user'
        });
    });

    // Uniqueness tests
    await runTest('Duplicate username', async () => {
        const user1 = new User({
            username: 'unique_user',
            email: 'unique1@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
        await user1.save();
        
        await expectValidationError('duplicate username', {
            username: 'unique_user',
            email: 'unique2@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
        
        await User.deleteOne({ _id: user1._id });
    });

    await runTest('Duplicate email', async () => {
        const user1 = new User({
            username: 'unique_user2',
            email: 'duplicate@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
        await user1.save();
        
        await expectValidationError('duplicate email', {
            username: 'unique_user3',
            email: 'duplicate@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
        
        await User.deleteOne({ _id: user1._id });
    });

    // CreatedAt default test
    await runTest('CreatedAt defaults to current timestamp', async () => {
        const user = new User({
            username: 'testuser26',
            email: 'test27@example.com',
            age: 25,
            role: 'user',
            accountStatus: 'active'
        });
        await user.save();
        
        if (!user.createdAt || !(user.createdAt instanceof Date)) {
            throw new Error('createdAt is not set or is not a Date');
        }
        
        const timeDiff = Math.abs(new Date() - user.createdAt);
        if (timeDiff > 5000) {
            throw new Error('createdAt is not close to current time');
        }
        
        await User.deleteOne({ _id: user._id });
    });

    // Print summary
    console.log('\n=== Test Summary ===');
    console.log(`Total Tests: ${testsPassed + testsFailed}`);
    console.log(`Passed: ${testsPassed}`);
    console.log(`Failed: ${testsFailed}`);

    if (testsFailed === 0) {
        console.log('\n✓ All tests passed!');
        return true;
    } else {
        console.log('\n✗ Some tests failed!');
        return false;
    }
}

// Main execution
async function main() {
    await connectDB();
    const allPassed = await runTests();
    await mongoose.connection.close();
    console.log('\nDatabase connection closed.');
    process.exit(allPassed ? 0 : 1);
}

main();