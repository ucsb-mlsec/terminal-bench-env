const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3000;

// Middleware to parse JSON bodies
app.use(bodyParser.json());

// Load users from users.json (simulating MongoDB collection)
let users = [];
try {
  const usersData = fs.readFileSync(path.join(__dirname, 'users.json'), 'utf8');
  users = JSON.parse(usersData);
} catch (error) {
  console.error('Error loading users:', error);
  users = [];
}

// Helper function to simulate MongoDB query evaluation
// This function is VULNERABLE to NoSQL injection
function evaluateQuery(user, query) {
  for (let key in query) {
    const queryValue = query[key];
    const userValue = user[key];
    
    // Check if query value is an object (potential operator injection)
    if (typeof queryValue === 'object' && queryValue !== null) {
      // Handle MongoDB operators - THIS IS THE VULNERABILITY
      for (let operator in queryValue) {
        if (operator === '$ne') {
          // Not equal operator
          if (userValue === queryValue[operator]) {
            return false;
          }
        } else if (operator === '$gt') {
          // Greater than operator
          if (!(userValue > queryValue[operator])) {
            return false;
          }
        } else if (operator === '$regex') {
          // Regex operator
          const regex = new RegExp(queryValue[operator]);
          if (!regex.test(userValue)) {
            return false;
          }
        }
      }
    } else {
      // Direct comparison
      if (userValue !== queryValue) {
        return false;
      }
    }
  }
  return true;
}

// Login endpoint - VULNERABLE TO NOSQL INJECTION
app.post('/login', (req, res) => {
  try {
    // Extract username and password from request body
    // NO VALIDATION OR SANITIZATION - THIS IS THE VULNERABILITY
    const { username, password } = req.body;
    
    // Simulate MongoDB findOne query
    // In real MongoDB: db.users.findOne({username: username, password: password})
    // This directly passes user input to the query, allowing injection
    const query = {
      username: username,
      password: password
    };
    
    console.log('Attempting authentication with query:', JSON.stringify(query));
    
    // Find user matching the query
    let authenticatedUser = null;
    for (let user of users) {
      if (evaluateQuery(user, query)) {
        authenticatedUser = user;
        break;
      }
    }
    
    // Check if authentication was successful
    if (authenticatedUser) {
      console.log('Authentication successful for user:', authenticatedUser.username);
      res.json({
        success: true,
        message: 'Authentication successful',
        user: {
          username: authenticatedUser.username,
          email: authenticatedUser.email
        }
      });
    } else {
      console.log('Authentication failed');
      res.status(401).json({
        success: false,
        message: 'Invalid username or password'
      });
    }
  } catch (error) {
    console.error('Error during authentication:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error'
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Authentication service running on port ${PORT}`);
  console.log(`Loaded ${users.length} users from database`);
});