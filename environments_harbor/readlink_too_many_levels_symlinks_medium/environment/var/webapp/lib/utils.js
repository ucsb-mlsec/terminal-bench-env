// Utility functions for the web application

/**
 * Format a date object into a readable string
 * @param {Date} date - The date to format
 * @returns {string} Formatted date string
 */
function formatDate(date) {
  const options = { year: 'numeric', month: 'long', day: 'numeric' };
  return date.toLocaleDateString('en-US', options);
}

/**
 * Validate email address format
 * @param {string} email - The email address to validate
 * @returns {boolean} True if valid email format
 */
function validateEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Generate a random unique identifier
 * @returns {string} Random ID string
 */
function randomId() {
  return Math.random().toString(36).substring(2, 15) + 
         Math.random().toString(36).substring(2, 15);
}

module.exports = {
  formatDate,
  validateEmail,
  randomId
};