const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;

// Middleware to parse JSON bodies for CSP reports
app.use(express.json({ type: 'application/csp-report' }));
app.use(express.json());

/**
 * CSP Middleware - CURRENT CONFIGURATION (PROBLEMATIC)
 * This is the restrictive CSP policy that is causing violations in production.
 * 
 * Current Policy Issues:
 * - script-src only allows 'self', blocking external analytics
 * - style-src only allows 'self', blocking inline styles
 * - connect-src only allows 'self', blocking external API calls
 * - font-src only allows 'self', blocking external font resources
 * 
 * This configuration needs to be analyzed and fixed based on violation logs.
 */
function cspMiddleware(req, res, next) {
  const cspPolicy = [
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self'",
    "connect-src 'self'",
    "font-src 'self'",
    "img-src 'self' data:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "report-uri /csp-report"
  ].join('; ');

  res.setHeader('Content-Security-Policy', cspPolicy);
  next();
}

// Apply CSP middleware to all routes
app.use(cspMiddleware);

/**
 * CSP Violation Report Endpoint
 * Receives violation reports from browsers and logs them to file
 * Format: JSON reports sent by browsers when CSP violations occur
 */
app.post('/csp-report', (req, res) => {
  const timestamp = new Date().toISOString();
  const violationReport = req.body;

  // Format the violation for logging
  const logEntry = {
    timestamp: timestamp,
    report: violationReport
  };

  // Append to CSP violations log file
  const logPath = path.join(__dirname, 'logs', 'csp-violations.log');
  const logLine = JSON.stringify(logEntry) + '\n';

  fs.appendFile(logPath, logLine, (err) => {
    if (err) {
      console.error('Error writing CSP violation to log:', err);
    }
  });

  // Always respond with 204 No Content for CSP reports
  res.status(204).end();
});

/**
 * Static file serving
 * Serves HTML, CSS, JS, and other static assets from the public directory
 */
app.use(express.static(path.join(__dirname, 'public')));

/**
 * Root route - serves the main application page
 */
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

/**
 * Health check endpoint
 */
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

/**
 * API endpoint example - for testing connect-src
 */
app.get('/api/data', (req, res) => {
  res.json({
    message: 'API data retrieved successfully',
    data: {
      items: ['item1', 'item2', 'item3']
    }
  });
});

/**
 * Error handling middleware
 */
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

/**
 * 404 handler for unknown routes
 */
app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

/**
 * Start the server
 */
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
  console.log(`CSP violation reports will be logged to: ${path.join(__dirname, 'logs', 'csp-violations.log')}`);
  console.log(`Current CSP Policy is RESTRICTIVE - monitor logs for violations`);
});

// Ensure logs directory exists
const logsDir = path.join(__dirname, 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}