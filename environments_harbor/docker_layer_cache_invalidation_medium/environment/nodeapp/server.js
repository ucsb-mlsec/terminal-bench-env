const express = require('express');
const path = require('path');
const app = express();
const PORT = 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files from public directory
app.use(express.static(path.join(__dirname, 'public')));

// Basic route
app.get('/', (req, res) => {
  res.send('Hello from Node.js app!');
});

// API status endpoint
app.get('/api/status', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString()
  });
});

// Mount routes from routes directory
try {
  const routes = require('./routes');
  app.use('/routes', routes);
} catch (err) {
  console.log('Routes module not found or error loading routes');
}

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});