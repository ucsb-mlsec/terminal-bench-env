const express = require('express');
const database = require('./database');

const app = express();

app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok' });
});

// Create payment endpoint
app.post('/api/payments', async (req, res) => {
  try {
    const { amount, currency, customerId } = req.body;
    const payment = await database.createPayment({ amount, currency, customerId });
    res.status(201).json({ success: true, payment });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get payment by ID endpoint
app.get('/api/payments/:id', async (req, res) => {
  try {
    const payment = await database.getPaymentById(req.params.id);
    if (!payment) {
      return res.status(404).json({ success: false, error: 'Payment not found' });
    }
    res.status(200).json({ success: true, payment });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Background health check that runs every 5 seconds
const healthCheckInterval = setInterval(() => {
  console.log('Health check running');
}, 5000);

function start(port) {
  const server = app.listen(port, () => {
    console.log(`Payment service listening on port ${port}`);
  });
  return server;
}

module.exports = { app, start };