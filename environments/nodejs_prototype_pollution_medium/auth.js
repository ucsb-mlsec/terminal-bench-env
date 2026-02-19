// Authentication Module for User Login
// Handles user authentication and authorization

const users = {
  'john': { username: 'john', password: 'pass123', isAdmin: false },
  'alice': { username: 'alice', password: 'secret456', isAdmin: false }
};

// Utility function to merge objects recursively
// WARNING: This implementation is vulnerable to prototype pollution
function deepMerge(target, source) {
  for (let key in source) {
    if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
      if (!target[key]) {
        target[key] = {};
      }
      deepMerge(target[key], source[key]);
    } else {
      target[key] = source[key];
    }
  }
  return target;
}

// Main authentication function
// Processes login requests and verifies user credentials
function authenticateUser(credentials) {
  // Validate input exists
  if (!credentials || !credentials.username || !credentials.password) {
    return { success: false, message: 'Missing credentials' };
  }

  // Create a new user object for the session
  let userSession = {};
  
  // Merge user-supplied credentials into the session object
  // VULNERABILITY: deepMerge doesn't sanitize __proto__ or constructor
  deepMerge(userSession, credentials);
  
  // Check if user exists in database
  const dbUser = users[credentials.username];
  
  if (!dbUser) {
    return { success: false, message: 'User not found' };
  }
  
  // Verify password
  if (dbUser.password !== credentials.password) {
    return { success: false, message: 'Invalid password' };
  }
  
  // Authorization check - verify admin privileges
  // VULNERABILITY: If Object.prototype.isAdmin is polluted, this check will pass
  if (userSession.isAdmin === true) {
    return { success: true, message: 'Admin access granted', role: 'admin', user: dbUser.username };
  }
  
  return { success: true, message: 'User authenticated', role: 'user', user: dbUser.username };
}

module.exports = { authenticateUser };