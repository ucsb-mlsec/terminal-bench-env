const { MongoClient } = require('mongodb');

const connectionString = 'mongodb://localhost:27017/myapp';
const dbName = 'myapp';

/**
 * Database module for user operations
 * Handles CRUD operations for user collection
 */

// Get user by ID
async function getUser(userId) {
    const client = new MongoClient(connectionString);
    await client.connect();
    
    const db = client.db(dbName);
    const users = db.collection('users');
    
    const user = await users.findOne({ _id: userId });
    
    return user;
}

// Save new user to database
async function saveUser(userData) {
    const client = new MongoClient(connectionString);
    await client.connect();
    
    const db = client.db(dbName);
    const users = db.collection('users');
    
    const result = await users.insertOne(userData);
    
    return result.insertedId;
}

// Update existing user
async function updateUser(userId, updateData) {
    const client = new MongoClient(connectionString);
    await client.connect();
    
    const db = client.db(dbName);
    const users = db.collection('users');
    
    const result = await users.updateOne(
        { _id: userId },
        { $set: updateData }
    );
    
    return result.modifiedCount > 0;
}

// Delete user by ID
async function deleteUser(userId) {
    const client = new MongoClient(connectionString);
    await client.connect();
    
    const db = client.db(dbName);
    const users = db.collection('users');
    
    const result = await users.deleteOne({ _id: userId });
    
    return result.deletedCount > 0;
}

module.exports = {
    getUser,
    saveUser,
    updateUser,
    deleteUser
};