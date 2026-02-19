const express = require('express');
const router = express.Router();
const db = require('../database');

// GET /users/:id - Retrieve a user by ID
router.get('/users/:id', async (req, res) => {
  try {
    const userId = req.params.id;
    const user = await db.getUser(userId);
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.json({ success: true, data: user });
  } catch (error) {
    console.error('Error fetching user:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /users - Create a new user
router.post('/users', async (req, res) => {
  try {
    const userData = req.body;
    const result = await db.saveUser(userData);
    
    res.status(201).json({ success: true, data: result });
  } catch (error) {
    console.error('Error creating user:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// PUT /users/:id - Update an existing user
router.put('/users/:id', async (req, res) => {
  try {
    const userId = req.params.id;
    const userData = req.body;
    const result = await db.updateUser(userId, userData);
    
    if (!result) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.json({ success: true, data: result });
  } catch (error) {
    console.error('Error updating user:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// DELETE /users/:id - Delete a user
router.delete('/users/:id', async (req, res) => {
  try {
    const userId = req.params.id;
    const result = await db.deleteUser(userId);
    
    if (!result) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.json({ success: true, message: 'User deleted successfully' });
  } catch (error) {
    console.error('Error deleting user:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;