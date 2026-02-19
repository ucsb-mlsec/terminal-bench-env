const express = require('express');
const router = express.Router();

// Home route
router.get('/', (req, res) => {
  res.send('Routes index');
});

// Users route
router.get('/users', (req, res) => {
  res.json([
    { id: 1, name: 'John Doe', email: 'john@example.com' },
    { id: 2, name: 'Jane Smith', email: 'jane@example.com' }
  ]);
});

// Health check route
router.get('/health', (req, res) => {
  res.json({ healthy: true });
});

module.exports = router;