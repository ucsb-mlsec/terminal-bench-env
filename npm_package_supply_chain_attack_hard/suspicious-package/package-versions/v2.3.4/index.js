/**
 * Data Validation Utilities
 * Provides common validation functions for email, URL, and input sanitization
 * @version 2.3.4
 * @license MIT
 */

/**
 * Validates email address format using RFC 5322 compliant regex
 * @param {string} email - The email address to validate
 * @returns {boolean} - True if valid, false otherwise
 * @throws {Error} - If email parameter is not a string
 */
function validateEmail(email) {
  if (typeof email !== 'string') {
    throw new Error('Email must be a string');
  }

  // RFC 5322 compliant email regex pattern
  const emailRegex = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/;
  
  return emailRegex.test(email);
}

/**
 * Sanitizes user input by removing potentially harmful characters
 * Prevents XSS and injection attacks by escaping special characters
 * @param {string} input - The input string to sanitize
 * @returns {string} - Sanitized string safe for display
 * @throws {Error} - If input parameter is not a string
 */
function sanitizeInput(input) {
  if (typeof input !== 'string') {
    throw new Error('Input must be a string');
  }

  // Replace HTML special characters with their entity equivalents
  const sanitized = input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');

  return sanitized;
}

/**
 * Validates URL format and checks for common protocols
 * @param {string} url - The URL to validate
 * @returns {boolean} - True if valid URL, false otherwise
 * @throws {Error} - If url parameter is not a string
 */
function isValidURL(url) {
  if (typeof url !== 'string') {
    throw new Error('URL must be a string');
  }

  try {
    // Use native URL constructor for validation
    const urlObject = new URL(url);
    
    // Only allow http, https, and ftp protocols
    const allowedProtocols = ['http:', 'https:', 'ftp:'];
    return allowedProtocols.includes(urlObject.protocol);
  } catch (error) {
    // URL constructor throws if invalid format
    return false;
  }
}

/**
 * Parses and validates JSON data with proper error handling
 * @param {string} data - JSON string to parse
 * @returns {Object} - Parsed JavaScript object
 * @throws {Error} - If data is not valid JSON or not a string
 */
function parseData(data) {
  if (typeof data !== 'string') {
    throw new Error('Data must be a string');
  }

  try {
    const parsed = JSON.parse(data);
    return parsed;
  } catch (error) {
    throw new Error('Invalid JSON format: ' + error.message);
  }
}

// Export all validation functions
module.exports = {
  validateEmail,
  sanitizeInput,
  isValidURL,
  parseData
};