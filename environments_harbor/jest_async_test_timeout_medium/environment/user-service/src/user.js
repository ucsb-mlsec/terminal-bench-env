// User service module - handles user management operations
// All functions work correctly - DO NOT MODIFY THIS FILE

const users = [
  { id: 1, name: 'Alice Johnson', email: 'alice@example.com' },
  { id: 2, name: 'Bob Smith', email: 'bob@example.com' },
  { id: 3, name: 'Charlie Brown', email: 'charlie@example.com' }
];

let nextId = 4;

/**
 * Fetches a user by ID - simulates async database call
 * @param {number} id - User ID
 * @returns {Promise<Object|null>} User object or null if not found
 */
function getUserById(id) {
  return new Promise((resolve) => {
    setTimeout(() => {
      const user = users.find(u => u.id === id);
      resolve(user || null);
    }, 20);
  });
}

/**
 * Creates a new user - simulates async database insert
 * @param {Object} userData - User data (name, email)
 * @returns {Promise<Object>} Created user with generated ID
 */
function createUser(userData) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!userData.name || !userData.email) {
        reject(new Error('Name and email are required'));
        return;
      }
      
      const newUser = {
        id: nextId++,
        name: userData.name,
        email: userData.email
      };
      users.push(newUser);
      resolve(newUser);
    }, 30);
  });
}

/**
 * Validates user data asynchronously
 * @param {Object} userData - User data to validate
 * @returns {Promise<boolean>} True if valid
 */
async function validateUser(userData) {
  return new Promise((resolve) => {
    setTimeout(() => {
      const isValid = userData && 
                     typeof userData.name === 'string' && 
                     userData.name.length > 0 &&
                     typeof userData.email === 'string' &&
                     userData.email.includes('@');
      resolve(isValid);
    }, 10);
  });
}

/**
 * Gets all users - simulates async database query
 * @returns {Promise<Array>} Array of all users
 */
function getAllUsers() {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve([...users]);
    }, 15);
  });
}

/**
 * Transforms user data by removing sensitive fields
 * @param {Object} user - User object
 * @returns {Object} Sanitized user object
 */
function sanitizeUser(user) {
  const { id, name } = user;
  return { id, name };
}

/**
 * Updates user data - simulates async database update
 * @param {number} id - User ID
 * @param {Object} updates - Fields to update
 * @returns {Promise<Object|null>} Updated user or null
 */
function updateUser(id, updates) {
  return new Promise((resolve) => {
    setTimeout(() => {
      const userIndex = users.findIndex(u => u.id === id);
      if (userIndex === -1) {
        resolve(null);
        return;
      }
      users[userIndex] = { ...users[userIndex], ...updates };
      resolve(users[userIndex]);
    }, 25);
  });
}

module.exports = {
  getUserById,
  createUser,
  validateUser,
  getAllUsers,
  sanitizeUser,
  updateUser
};