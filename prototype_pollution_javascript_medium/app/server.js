const express = require('express');
const bodyParser = require('body-parser');
const preferences = require('./preferences');

const app = express();
const PORT = 3000;

app.use(bodyParser.json());

app.post('/api/preferences/:userId', async (req, res) => {
  try {
    const userId = req.params.userId;
    const updates = req.body;
    
    const updatedPreferences = await preferences.updatePreferences(userId, updates);
    res.status(200).json(updatedPreferences);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/preferences/:userId', async (req, res) => {
  try {
    const userId = req.params.userId;
    const userPreferences = await preferences.getPreferences(userId);
    res.status(200).json(userPreferences);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}

module.exports = app;