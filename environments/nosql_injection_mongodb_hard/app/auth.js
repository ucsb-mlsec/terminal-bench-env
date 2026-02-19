const express = require('express');
const { MongoClient } = require('mongodb');
const bcrypt = require('bcrypt');

const app = express();
app.use(express.json());

let db;
const mongoUrl = process.env.MONGO_URL || 'mongodb://localhost:27017';
const dbName = 'authService';

// Initialize MongoDB connection
MongoClient.connect(mongoUrl, { useUnifiedTopology: true })
  .then(client => {
    db = client.db(dbName);
    console.log('Connected to MongoDB');
  })
  .catch(err => console.error('MongoDB connection error:', err));

// Login endpoint - authenticates user credentials
// POST /api/login
// Body: { username: string, password: string }
app.post('/api/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    
    // Find user by username and password
    const user = await db.collection('users').findOne({
      username: username,
      password: password
    });

    if (user) {
      // Generate session token
      const sessionToken = generateToken(user._id);
      return res.json({ 
        success: true, 
        token: sessionToken,
        userId: user._id 
      });
    } else {
      return res.status(401).json({ 
        success: false, 
        message: 'Invalid credentials' 
      });
    }
  } catch (error) {
    console.error('Login error:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Internal server error' 
    });
  }
});

// Password reset function - allows users to reset their password
// POST /api/reset-password
// Body: { resetToken: string, newPassword: string, query: object }
app.post('/api/reset-password', async (req, res) => {
  try {
    const { resetToken, newPassword, query } = req.body;

    // Verify reset token is present
    if (!resetToken || !newPassword) {
      return res.status(400).json({ 
        success: false, 
        message: 'Missing required fields' 
      });
    }

    // Find user with valid reset token and additional query conditions
    const userQuery = query || {};
    userQuery.resetToken = resetToken;
    userQuery.resetTokenExpiry = { $gt: new Date() };

    const user = await db.collection('users').findOne(userQuery);

    if (!user) {
      return res.status(400).json({ 
        success: false, 
        message: 'Invalid or expired reset token' 
      });
    }

    // Hash new password and update user
    const hashedPassword = await bcrypt.hash(newPassword, 10);
    await db.collection('users').updateOne(
      { _id: user._id },
      { 
        $set: { password: hashedPassword },
        $unset: { resetToken: '', resetTokenExpiry: '' }
      }
    );

    return res.json({ 
      success: true, 
      message: 'Password updated successfully' 
    });
  } catch (error) {
    console.error('Password reset error:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Internal server error' 
    });
  }
});

// User lookup function - retrieves user profile information
// GET /api/user?email=user@example.com
app.get('/api/user', async (req, res) => {
  try {
    const { email } = req.query;

    if (!email) {
      return res.status(400).json({ 
        success: false, 
        message: 'Email parameter is required' 
      });
    }

    // Find user by email
    const user = await db.collection('users').findOne({ 
      email: email 
    });

    if (user) {
      // Remove sensitive fields before returning
      delete user.password;
      delete user.resetToken;
      
      return res.json({ 
        success: true, 
        user: user 
      });
    } else {
      return res.status(404).json({ 
        success: false, 
        message: 'User not found' 
      });
    }
  } catch (error) {
    console.error('User lookup error:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Internal server error' 
    });
  }
});

// Helper function to generate session tokens
function generateToken(userId) {
  return Buffer.from(`${userId}:${Date.now()}`).toString('base64');
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Authentication service running on port ${PORT}`);
});

module.exports = app;