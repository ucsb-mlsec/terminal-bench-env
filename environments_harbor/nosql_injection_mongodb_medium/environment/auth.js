const mongoose = require('mongoose');

// MongoDB connection setup
mongoose.connect('mongodb://localhost:27017/authdb', {
  useNewUrlParser: true,
  useUnifiedTopology: true
});

const userSchema = new mongoose.Schema({
  username: String,
  password: String,
  email: String,
  role: String
});

const User = mongoose.model('User', userSchema);

/**
 * Authenticates a user by verifying their credentials against the database
 * @param {string} username - The username provided by the user
 * @param {string} password - The password provided by the user
 * @returns {Promise} Resolves with user object if authenticated, null otherwise
 */
async function authenticateUser(username, password) {
  try {
    // Query the database to find matching user credentials
    const user = await User.findOne({
      username: username,
      password: password
    });

    if (user) {
      console.log('Authentication successful for user:', username);
      return user;
    } else {
      console.log('Authentication failed: Invalid credentials');
      return null;
    }
  } catch (error) {
    console.error('Authentication error:', error);
    throw error;
  }
}

/**
 * Alternative login method using native MongoDB driver
 * @param {string} email - User email address
 * @param {string} password - User password
 */
async function loginByEmail(email, password) {
  const db = mongoose.connection.db;
  const usersCollection = db.collection('users');
  
  // Find user by email and password
  const result = await usersCollection.findOne({
    email: email,
    password: password
  });
  
  return result;
}

module.exports = {
  authenticateUser,
  loginByEmail
};