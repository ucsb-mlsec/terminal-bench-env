const express = require('express');
const fs = require('fs');
const path = require('path');

// Read weather data from JSON file
const dataPath = path.join(__dirname, 'data', 'observations.json');
const rawData = fs.readFileSync(dataPath, 'utf8');
const weatherData = JSON.parse(rawData);

// Create Express application
const app = express();
const PORT = 3000;

// Define GET endpoint for weather data
app.get('/weather', (req, res) => {
  res.json(weatherData);
});

// Start the server
app.listen(PORT, () => {
  console.log(`Weather service is running on port ${PORT}`);
});